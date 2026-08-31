"""结构化任务规划。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.llm import LLM
from core.schemas import PlanStep, Skill, Task
from core.token_utils import estimate_tokens

from .schema_validation import validate_arguments, validate_last_output_references
from .disclosure_policy import SignalDisclosurePolicy


@dataclass
class PlanningResult:
    """规划结果及分阶段披露、校验和调用成本证据。"""

    steps: list[PlanStep] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    selection_evidence: list[dict] = field(default_factory=list)
    selection_context_tokens: int = 0
    planning_context_tokens: int = 0
    validation_errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    planner_calls: int = 0
    disclosure_level: str = "full"
    requested_disclosure_levels: list[str] = field(default_factory=list)
    disclosure_reasons: list[str] = field(default_factory=list)
    escalation_reasons: list[str] = field(default_factory=list)
    disclosure_signals: dict = field(default_factory=dict)


class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.signal_disclosure_policy = SignalDisclosurePolicy()

    def plan(
        self,
        task: Task,
        skill_context: str,
        allowed_skill_names: set[str],
    ) -> list[PlanStep]:
        """生成经过名称校验的执行计划。"""
        input_context = self._input_context(task)
        prompt = (
            f"任务：{task.instruction}\n\n"
            f"{input_context}"
            f"可用技能：\n{skill_context}\n\n"
            "选择完成任务所需的最少技能，并严格按照执行顺序输出。"
            "skill_name 必须来自可用技能，arguments 必须符合参数 Schema。"
            "后续步骤引用上一步输出时，参数值使用 $last_output；"
            "引用原任务时使用 $task。"
            "引用任务输入时使用 $input.<字段名>。"
            '只输出 JSON：{"plan": [{"skill_name": "名称", '
            '"arguments": {}, "reason": "原因"}]}。'
        )
        data = self.llm.generate_json([{"role": "user", "content": prompt}])
        if not isinstance(data, dict) or not isinstance(data.get("plan"), list):
            return []

        steps: list[PlanStep] = []
        for item in data["plan"]:
            if not isinstance(item, dict):
                continue
            name = item.get("skill_name")
            arguments = item.get("arguments", {})
            if name not in allowed_skill_names or not isinstance(arguments, dict):
                continue
            steps.append(PlanStep(
                skill_name=name,
                arguments=arguments,
                reason=str(item.get("reason", "")),
            ))
        return steps

    def plan_two_stage(
        self,
        task: Task,
        selection_context: str,
        available_skills: list[Skill],
        max_argument_repairs: int = 1,
        disclosure_level: str = "full",
    ) -> PlanningResult:
        """先选择 Skill，再按指定层级披露所选 Skill 并校验参数。"""
        if max_argument_repairs not in {0, 1}:
            raise ValueError("max_argument_repairs 只能是 0 或 1")
        if disclosure_level not in {
            "brief", "schema", "full", "adaptive", "adaptive_signals"
        }:
            raise ValueError(
                "disclosure_level 必须是 brief、schema、full、adaptive 或 "
                "adaptive_signals"
            )
        by_id = {skill.id: skill for skill in available_skills}
        if disclosure_level == "adaptive":
            disclosure_instruction = (
                "为每个选择标最低信息层级：单一原子操作且具名任务输入可一对一"
                "直传时选 brief；多步数据流或字段映射不明时选 schema；只有任务"
                "明确依赖副作用、覆盖/追加、格式保真等非 Schema 行为语义时选 full。"
                "不得为了保险升档。"
            )
            output_contract = (
                '只输出 JSON：{"selections": [{"requirement": "原子操作", '
                '"skill_id": "Skill ID", "information_need": '
                '"brief|schema|full"}]}。'
            )
        else:
            disclosure_instruction = ""
            output_contract = (
                '只输出 JSON：{"selections": [{"requirement": "原子操作", '
                '"skill_id": "Skill ID"}]}。'
            )
        selection_prompt = (
            f"任务：{task.instruction}\n\n"
            f"{self._input_context(task)}"
            f"候选技能概览：\n{selection_context}\n\n"
            "先把任务拆成必须完成的原子操作，尤其不能遗漏“先、再、然后”"
            "连接的后续操作。为每个原子操作选择一个最匹配的 Skill；"
            "最终结果必须覆盖任务要求的全部转换，而不只是第一个中间结果。"
            "此阶段不要生成参数。skill_id 必须来自方括号中的候选 ID。"
            f"{disclosure_instruction}{output_contract}"
        )
        selected_data = self.llm.generate_json(
            [{"role": "user", "content": selection_prompt}]
        )
        selected_ids, selection_evidence = self._selected_ids(
            selected_data,
            set(by_id),
        )
        selected_skills = [by_id[skill_id] for skill_id in selected_ids]
        requested_levels = [
            str(item.get("information_need", "brief"))
            for item in selection_evidence
        ]
        disclosure_reasons = [
            str(item.get("need_reason", ""))
            for item in selection_evidence
            if item.get("need_reason")
        ]
        disclosure_signals: dict = {}
        if disclosure_level == "adaptive":
            resolved_level = self._highest_disclosure_level(requested_levels)
        elif disclosure_level == "adaptive_signals":
            decision = self.signal_disclosure_policy.decide(task, selected_skills)
            resolved_level = decision.level
            requested_levels = [decision.level]
            disclosure_reasons = [decision.reason]
            disclosure_signals = decision.signals
        else:
            resolved_level = disclosure_level
        if not selected_skills:
            return PlanningResult(
                selection_context_tokens=estimate_tokens(selection_context),
                selection_evidence=selection_evidence,
                planner_calls=1,
                disclosure_level=resolved_level,
                requested_disclosure_levels=requested_levels,
                disclosure_reasons=disclosure_reasons,
                disclosure_signals=disclosure_signals,
            )

        planning_context = "\n\n".join(
            skill.to_prompt(level=resolved_level) for skill in selected_skills
        )
        selected_names = [skill.name for skill in selected_skills]
        plan_prompt = self._planning_prompt(task, planning_context, resolved_level)
        plan_data = self.llm.generate_json([{"role": "user", "content": plan_prompt}])
        steps = self._parse_plan(plan_data, set(selected_names))
        errors = self._validation_errors(steps, selected_skills, task)
        repair_attempts = 0
        calls = 2
        escalation_reasons: list[str] = []
        if errors and max_argument_repairs == 1:
            repair_attempts = 1
            calls += 1
            if disclosure_level in {"adaptive", "adaptive_signals"} and (
                resolved_level == "brief"
            ):
                resolved_level = "schema"
                escalation_reasons.append("hidden_schema_validation_failed")
                planning_context = "\n\n".join(
                    skill.to_prompt(level=resolved_level)
                    for skill in selected_skills
                )
            repair_prompt = (
                f"任务：{task.instruction}\n\n"
                f"{self._input_context(task)}"
                f"已选技能定义（披露层级={resolved_level}）：\n"
                f"{planning_context}\n\n"
                f"无效计划：{json.dumps(plan_data, ensure_ascii=False)}\n\n"
                f"Schema 校验错误：\n- " + "\n- ".join(errors) + "\n\n"
                "只在已选技能范围内修复参数、顺序或计划结构，不得选择其他技能。"
                "已选技能只是可用上限，不要求全部调用；如果某一步冗余或其输入与"
                "$last_output 不兼容，可以删除该步，并尽量通过上游技能参数（例如"
                "SQL 的 ORDER BY）直接满足对应要求。"
                '只输出 JSON：{"plan": [{"skill_name": "名称", '
                '"arguments": {}, "reason": "原因"}]}。'
            )
            repaired_data = self.llm.generate_json(
                [{"role": "user", "content": repair_prompt}]
            )
            steps = self._parse_plan(repaired_data, set(selected_names))
            errors = self._validation_errors(steps, selected_skills, task)

        return PlanningResult(
            steps=steps if not errors else [],
            selected_skill_ids=[skill.id for skill in selected_skills],
            selection_evidence=selection_evidence,
            selection_context_tokens=estimate_tokens(selection_context),
            planning_context_tokens=estimate_tokens(planning_context),
            validation_errors=errors,
            repair_attempts=repair_attempts,
            planner_calls=calls,
            disclosure_level=resolved_level,
            requested_disclosure_levels=requested_levels,
            disclosure_reasons=disclosure_reasons,
            escalation_reasons=escalation_reasons,
            disclosure_signals=disclosure_signals,
        )

    @staticmethod
    def _selected_ids(
        data: object,
        allowed_ids: set[str],
    ) -> tuple[list[str], list[dict]]:
        if not isinstance(data, dict):
            return [], []
        selections = data.get("selections")
        evidence: list[dict] = []
        skill_ids: list[str] = []
        if isinstance(selections, list):
            for item in selections:
                if not isinstance(item, dict):
                    continue
                skill_id = item.get("skill_id")
                if isinstance(skill_id, str) and skill_id in allowed_ids:
                    evidence.append({
                        "requirement": str(item.get("requirement", "")),
                        "skill_id": skill_id,
                        "information_need": (
                            item.get("information_need")
                            if item.get("information_need")
                            in {"brief", "schema", "full"}
                            else "brief"
                        ),
                        "need_reason": str(item.get("need_reason", "")),
                    })
                    if skill_id not in skill_ids:
                        skill_ids.append(skill_id)
            return skill_ids, evidence
        legacy_ids = data.get("skill_ids")
        if not isinstance(legacy_ids, list):
            return [], []
        for item in legacy_ids:
            if isinstance(item, str) and item in allowed_ids and item not in skill_ids:
                skill_ids.append(item)
                evidence.append({
                    "requirement": "",
                    "skill_id": item,
                    "information_need": "brief",
                    "need_reason": "legacy selection response",
                })
        return skill_ids, evidence

    @staticmethod
    def _highest_disclosure_level(levels: list[str]) -> str:
        ranking = {"brief": 0, "schema": 1, "full": 2}
        valid = [level for level in levels if level in ranking]
        return max(valid, key=ranking.get) if valid else "brief"

    @staticmethod
    def _parse_plan(data: object, allowed_names: set[str]) -> list[PlanStep]:
        if not isinstance(data, dict) or not isinstance(data.get("plan"), list):
            return []
        steps: list[PlanStep] = []
        for item in data["plan"]:
            if not isinstance(item, dict):
                continue
            name = item.get("skill_name")
            arguments = item.get("arguments", {})
            if name not in allowed_names or not isinstance(arguments, dict):
                continue
            steps.append(PlanStep(
                skill_name=name,
                arguments=arguments,
                reason=str(item.get("reason", "")),
            ))
        return steps

    @staticmethod
    def _validation_errors(
        steps: list[PlanStep],
        selected_skills: list[Skill],
        task: Task,
    ) -> list[str]:
        if not steps:
            return ["计划为空或格式无效"]
        by_name = {skill.name: skill for skill in selected_skills}
        errors: list[str] = []
        for index, step in enumerate(steps, 1):
            skill = by_name[step.skill_name]
            errors.extend(
                f"步骤 {index} ({step.skill_name}): {error}"
                for error in validate_arguments(
                    step.arguments,
                    skill.parameters,
                    task.instruction,
                    task.inputs,
                )
            )
            previous_returns = None
            if index > 1:
                previous_returns = by_name[steps[index - 2].skill_name].returns
            errors.extend(
                f"步骤 {index} ({step.skill_name}): {error}"
                for error in validate_last_output_references(
                    step.arguments,
                    skill.parameters,
                    previous_returns,
                )
            )
        return errors

    @staticmethod
    def _planning_prompt(
        task: Task,
        planning_context: str,
        disclosure_level: str = "full",
    ) -> str:
        return (
            f"任务：{task.instruction}\n\n"
            f"{Planner._input_context(task)}"
            f"已选技能定义（披露层级={disclosure_level}）：\n"
            f"{planning_context}\n\n"
            "使用这些技能生成最短可执行计划，严格遵守参数 Schema。"
            "已选技能是可用上限，不要求全部调用；能通过一个技能的参数完成的排序、"
            "筛选或格式要求，不要再增加不兼容的下游步骤。"
            "后续步骤引用上一步输出时使用 $last_output；"
            "引用原任务文本时使用 $task。"
            "引用任务输入时使用 $input.<字段名>。"
            '只输出 JSON：{"plan": [{"skill_name": "名称", '
            '"arguments": {}, "reason": "原因"}]}。'
        )

    @staticmethod
    def _input_context(task: Task) -> str:
        if not task.inputs:
            return ""
        return (
            "任务输入（可通过 $input.<字段名> 精确引用）：\n"
            f"{json.dumps(task.inputs, ensure_ascii=False, sort_keys=True)}\n\n"
        )
