"""Agent 执行流程。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.llm import LLM
from core.schemas import RetrievalResult, Task
from core.token_utils import estimate_tokens
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
    ):
        self.llm = llm
        self.organizer = organizer
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        self.skill_handlers = skill_handlers or {}
        self.planner = Planner(llm)
        self.reflection = Reflection(llm)

    def run(self, task: Task, retrieval: RetrievalResult) -> dict:
        """执行任务并返回结构化轨迹和计量数据。"""
        usage_before = self.llm.usage
        skills = retrieval.skills
        if not skills:
            return self._result(
                trajectory=[],
                success=False,
                answer="无可用技能",
                skill_context_tokens=0,
                execution_steps=0,
                usage_before=usage_before,
            )

        skill_context = self.organizer.organize(skills, task)
        state = AgentStateManager(task)
        executor = Executor(skills, self.skill_handlers)
        plan = self.planner.plan(task, skill_context, {skill.name for skill in skills})
        state.state.plan = plan
        state.log(
            "plan",
            " -> ".join(step.skill_name for step in plan) or "未生成有效计划",
            plan=[step.__dict__ for step in plan],
        )

        outputs: list[str] = []
        call_results = []
        for index, plan_step in enumerate(plan[: self.max_steps], 1):
            arguments = self._resolve_arguments(
                plan_step.arguments,
                task_instruction=task.instruction,
                last_output=outputs[-1] if outputs else "",
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
        answer = "\n".join(outputs)
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
            skill_context_tokens=estimate_tokens(skill_context),
            execution_steps=1 + len(call_results),
            usage_before=usage_before,
        )

    @classmethod
    def _resolve_arguments(
        cls,
        value: Any,
        task_instruction: str,
        last_output: str,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._resolve_arguments(item, task_instruction, last_output)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._resolve_arguments(item, task_instruction, last_output)
                for item in value
            ]
        if value == "$last_output":
            return last_output
        if value == "$task":
            return task_instruction
        return value

    def _result(
        self,
        trajectory: list[dict],
        success: bool,
        answer: str,
        skill_context_tokens: int,
        execution_steps: int,
        usage_before: dict[str, int],
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
        return {
            "trajectory": trajectory,
            "success": success,
            "answer": answer,
            "selected_skill_ids": selected_ids,
            "skill_calls": len(selected_ids),
            "skill_context_tokens": skill_context_tokens,
            "token_usage": token_usage,
            "execution_steps": execution_steps,
            "execution_time_ms": execution_time_ms,
        }
