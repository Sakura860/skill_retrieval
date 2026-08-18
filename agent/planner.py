"""结构化任务规划。"""
from __future__ import annotations

from core.llm import LLM
from core.schemas import PlanStep, Task


class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def plan(
        self,
        task: Task,
        skill_context: str,
        allowed_skill_names: set[str],
    ) -> list[PlanStep]:
        """生成经过名称校验的执行计划。"""
        prompt = (
            f"任务：{task.instruction}\n\n"
            f"可用技能：\n{skill_context}\n\n"
            "选择完成任务所需的最少技能，并严格按照执行顺序输出。"
            "skill_name 必须来自可用技能，arguments 必须符合参数 Schema。"
            "后续步骤引用上一步输出时，参数值使用 $last_output；"
            "引用原任务时使用 $task。"
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
