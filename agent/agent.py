"""Agent 执行流程。"""
from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from core.llm import LLM
from core.schemas import RetrievalResult, Task
from execution.environment import TaskEnvironment
from execution.registry import SkillRegistry
from organization.base import BaseOrganizer

from .executor import Executor
from .planner import Planner
from .reflection import Reflection
from .state import AgentStateManager


class Agent:
    def __init__(
        self,
        llm: LLM,
        organizer: BaseOrganizer,
        max_steps: int = 10,
        enable_reflection: bool = True,
        skill_handlers: dict[str, Callable[..., Any]] | None = None,
        skill_registry: SkillRegistry | None = None,
        planner_mode: str = "one_stage",
        max_argument_repairs: int = 1,
        planner_disclosure_level: str = "full",
    ):
        if planner_mode not in {"one_stage", "two_stage"}:
            raise ValueError("planner_mode 必须是 one_stage 或 two_stage")
        if max_argument_repairs not in {0, 1}:
            raise ValueError("max_argument_repairs 只能是 0 或 1")
        if planner_disclosure_level not in {
            "brief", "schema", "full", "adaptive", "adaptive_signals"
        }:
            raise ValueError(
                "planner_disclosure_level 必须是 brief、schema、full、adaptive "
                "或 adaptive_signals"
            )
        self.llm = llm
        self.organizer = organizer
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        self.skill_handlers = skill_handlers or {}
        self.skill_registry = skill_registry
        self.planner_mode = planner_mode
        self.max_argument_repairs = max_argument_repairs
        self.planner_disclosure_level = planner_disclosure_level
        self.planner = Planner(llm)
        self.reflection = Reflection(llm)

    def run(
        self,
        task: Task,
        retrieval: RetrievalResult,
        environment: TaskEnvironment | None = None,
        context_budget_tokens: int | None = None,
    ) -> dict:
        """执行任务并返回结构化轨迹和计量数据。"""
        run_started = time.perf_counter()
        usage_before = self.llm.usage
        existing_llm_calls = getattr(self.llm, "calls", [])
        llm_call_start = len(existing_llm_calls) if isinstance(
            existing_llm_calls, list
        ) else 0
        initial_state = (
            environment.initial_state_copy() if environment is not None else None
        )
        skills = retrieval.skills
        if not skills:
            return self._result(
                trajectory=[],
                success=False,
                answer="无可用技能",
                skill_context_tokens=0,
                execution_steps=0,
                usage_before=usage_before,
                initial_state=initial_state,
                final_state=(environment.snapshot() if environment else None),
                context_budget_tokens=context_budget_tokens,
                planner_mode=self.planner_mode,
                planner_disclosure_level=self.planner_disclosure_level,
                run_started=run_started,
                llm_call_start=llm_call_start,
            )

        organized = self.organizer.organize_context(
            skills,
            task,
            context_budget_tokens=context_budget_tokens,
        )
        skills = organized.resolved_skills or skills
        skill_context = organized.text
        state = AgentStateManager(task)
        handlers = dict(self.skill_handlers)
        if self.skill_registry is not None:
            if environment is None:
                raise RuntimeError("使用 SkillRegistry 时必须提供 TaskEnvironment")
            handlers.update(self.skill_registry.handlers_for(environment))
        executor = Executor(skills, handlers)
        exposed_ids = set(organized.exposed_skill_ids)
        exposed_skills = [skill for skill in skills if skill.id in exposed_ids]
        allowed_names = {skill.name for skill in exposed_skills}
        selection_context_tokens = 0
        planning_context_tokens = 0
        planner_calls = 1
        repair_attempts = 0
        validation_errors: list[str] = []
        selection_evidence: list[dict] = []
        requested_disclosure_levels: list[str] = []
        disclosure_reasons: list[str] = []
        escalation_reasons: list[str] = []
        resolved_disclosure_level = self.planner_disclosure_level
        disclosure_signals: dict = {}
        disclosed_ids = list(organized.detailed_skill_ids)
        if self.planner_mode == "two_stage":
            selection_context = "\n".join(
                f"- [{skill.id}] {skill.to_prompt(detailed=False)}"
                for skill in exposed_skills
            )
            planning = self.planner.plan_two_stage(
                task,
                selection_context,
                exposed_skills,
                max_argument_repairs=self.max_argument_repairs,
                disclosure_level=self.planner_disclosure_level,
            )
            plan = planning.steps
            disclosed_ids = planning.selected_skill_ids
            selection_context_tokens = planning.selection_context_tokens
            planning_context_tokens = planning.planning_context_tokens
            planner_calls = planning.planner_calls
            repair_attempts = planning.repair_attempts
            validation_errors = planning.validation_errors
            selection_evidence = planning.selection_evidence
            requested_disclosure_levels = planning.requested_disclosure_levels
            disclosure_reasons = planning.disclosure_reasons
            escalation_reasons = planning.escalation_reasons
            resolved_disclosure_level = planning.disclosure_level
            disclosure_signals = planning.disclosure_signals
            skill_context_tokens = (
                selection_context_tokens + planning_context_tokens
            )
        else:
            plan = self.planner.plan(task, skill_context, allowed_names)
            selection_context_tokens = organized.token_count
            skill_context_tokens = organized.token_count
        state.state.plan = plan
        state.log(
            "plan",
            " -> ".join(step.skill_name for step in plan) or "未生成有效计划",
            plan=[step.__dict__ for step in plan],
            planner_mode=self.planner_mode,
            selected_for_schema=disclosed_ids,
            selection_evidence=selection_evidence,
            requested_disclosure_levels=requested_disclosure_levels,
            disclosure_reasons=disclosure_reasons,
            escalation_reasons=escalation_reasons,
            resolved_disclosure_level=resolved_disclosure_level,
            disclosure_signals=disclosure_signals,
            validation_errors=validation_errors,
            repair_attempts=repair_attempts,
        )

        outputs: list[str] = []
        call_results = []
        for index, plan_step in enumerate(plan[: self.max_steps], 1):
            arguments = self._resolve_arguments(
                plan_step.arguments,
                task_instruction=task.instruction,
                last_output=outputs[-1] if outputs else "",
                task_inputs=task.inputs,
            )
            call = executor.call(plan_step.skill_name, arguments)
            call_results.append(call)
            content = call.output if call.success else call.error
            state.log(
                "action",
                content,
                index=index,
                skill_id=call.skill_id,
                skill_name=call.skill_name,
                arguments=arguments,
                success=call.success,
                duration_ms=call.duration_ms,
            )
            if call.output:
                outputs.append(call.output)
            if not call.success:
                break

        success = bool(call_results) and all(call.success for call in call_results)
        answer = outputs[-1] if outputs else ""
        if not success and call_results:
            answer = call_results[-1].error

        state.state.finished = True
        if self.enable_reflection:
            reflection = self.reflection.reflect(
                task.instruction,
                state.state.to_prompt_context(),
                success,
            )
            state.state.reflections.append(reflection)
            state.log("reflection", reflection)

        return self._result(
            trajectory=state.state.history,
            success=success,
            answer=answer,
            skill_context_tokens=skill_context_tokens,
            execution_steps=1 + len(call_results),
            usage_before=usage_before,
            initial_state=initial_state,
            final_state=(environment.snapshot() if environment else None),
            exposed_skill_ids=organized.exposed_skill_ids,
            detailed_skill_ids=disclosed_ids,
            truncated_skill_ids=organized.truncated_skill_ids,
            context_budget_tokens=organized.context_budget_tokens,
            planner_mode=self.planner_mode,
            planner_disclosure_level=self.planner_disclosure_level,
            planner_calls=planner_calls,
            repair_attempts=repair_attempts,
            validation_errors=validation_errors,
            selection_context_tokens=selection_context_tokens,
            planning_context_tokens=planning_context_tokens,
            selection_evidence=selection_evidence,
            requested_disclosure_levels=requested_disclosure_levels,
            disclosure_reasons=disclosure_reasons,
            escalation_reasons=escalation_reasons,
            resolved_disclosure_level=resolved_disclosure_level,
            disclosure_signals=disclosure_signals,
            added_skill_ids=organized.added_skill_ids,
            graph_issues=organized.graph_issues,
            failure_reason=(
                "schema_validation_failed" if validation_errors else None
            ),
            run_started=run_started,
            llm_call_start=llm_call_start,
        )

    @classmethod
    def _resolve_arguments(
        cls,
        value: Any,
        task_instruction: str,
        last_output: str,
        task_inputs: dict[str, Any] | None = None,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._resolve_arguments(
                    item, task_instruction, last_output, task_inputs
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._resolve_arguments(
                    item, task_instruction, last_output, task_inputs
                )
                for item in value
            ]
        if value == "$last_output":
            return last_output
        if value == "$task":
            return task_instruction
        if isinstance(value, str) and value.startswith("$input."):
            key = value[len("$input."):]
            if task_inputs is not None and key in task_inputs:
                return task_inputs[key]
        return value

    def _result(
        self,
        trajectory: list[dict],
        success: bool,
        answer: str,
        skill_context_tokens: int,
        execution_steps: int,
        usage_before: dict[str, int],
        initial_state: Any = None,
        final_state: Any = None,
        exposed_skill_ids: list[str] | None = None,
        detailed_skill_ids: list[str] | None = None,
        truncated_skill_ids: list[str] | None = None,
        context_budget_tokens: int | None = None,
        planner_mode: str = "one_stage",
        planner_disclosure_level: str = "full",
        planner_calls: int = 0,
        repair_attempts: int = 0,
        validation_errors: list[str] | None = None,
        selection_context_tokens: int = 0,
        planning_context_tokens: int = 0,
        failure_reason: str | None = None,
        selection_evidence: list[dict] | None = None,
        added_skill_ids: list[str] | None = None,
        graph_issues: list[str] | None = None,
        run_started: float | None = None,
        llm_call_start: int = 0,
        requested_disclosure_levels: list[str] | None = None,
        disclosure_reasons: list[str] | None = None,
        escalation_reasons: list[str] | None = None,
        resolved_disclosure_level: str | None = None,
        disclosure_signals: dict | None = None,
    ) -> dict:
        usage_after = self.llm.usage
        token_usage = {
            key: usage_after[key] - usage_before.get(key, 0)
            for key in usage_after
        }
        selected_ids = [
            item.get("skill_id", "")
            for item in trajectory
            if item.get("step") == "action" and item.get("skill_id")
        ]
        execution_time_ms = sum(
            float(item.get("duration_ms", 0.0))
            for item in trajectory
            if item.get("step") == "action"
        )
        all_llm_calls = getattr(self.llm, "calls", [])
        if not isinstance(all_llm_calls, list):
            all_llm_calls = []
        llm_calls = list(all_llm_calls[llm_call_start:])
        llm_time_ms = sum(float(item.get("duration_ms", 0.0)) for item in llm_calls)
        end_to_end_time_ms = (
            (time.perf_counter() - run_started) * 1000
            if run_started is not None
            else execution_time_ms + llm_time_ms
        )
        return {
            "trajectory": trajectory,
            "success": success,
            "answer": answer,
            "failure_reason": failure_reason,
            "selected_skill_ids": selected_ids,
            "skill_calls": len(selected_ids),
            "skill_context_tokens": skill_context_tokens,
            "context_budget_tokens": context_budget_tokens,
            "planner_mode": planner_mode,
            "planner_disclosure_level": planner_disclosure_level,
            "resolved_disclosure_level": (
                resolved_disclosure_level or planner_disclosure_level
            ),
            "requested_disclosure_levels": list(
                requested_disclosure_levels or []
            ),
            "disclosure_reasons": list(disclosure_reasons or []),
            "escalation_reasons": list(escalation_reasons or []),
            "disclosure_signals": dict(disclosure_signals or {}),
            "planner_calls": planner_calls,
            "repair_attempts": repair_attempts,
            "validation_errors": list(validation_errors or []),
            "selection_evidence": list(selection_evidence or []),
            "added_skill_ids": list(added_skill_ids or []),
            "graph_issues": list(graph_issues or []),
            "selection_context_tokens": selection_context_tokens,
            "planning_context_tokens": planning_context_tokens,
            "exposed_skill_ids": list(exposed_skill_ids or []),
            "detailed_skill_ids": list(detailed_skill_ids or []),
            "truncated_skill_ids": list(truncated_skill_ids or []),
            "token_usage": token_usage,
            "execution_steps": execution_steps,
            "execution_time_ms": execution_time_ms,
            "llm_time_ms": llm_time_ms,
            "end_to_end_time_ms": end_to_end_time_ms,
            "llm_calls": llm_calls,
            "initial_state": initial_state,
            "final_state": final_state,
        }
