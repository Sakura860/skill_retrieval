"""受控 Skill 执行器。"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from core.schemas import Skill, SkillCallResult

SkillHandler = Callable[..., Any]


class Executor:
    """通过显式注册的处理函数执行 Skill。"""

    def __init__(
        self,
        skills: list[Skill],
        handlers: dict[str, SkillHandler] | None = None,
    ):
        self._by_name = {skill.name: skill for skill in skills}
        self._handlers = handlers or {}

    def call(self, skill_name: str, arguments: dict | None = None) -> SkillCallResult:
        """校验名称和参数后执行 Skill。"""
        started = time.perf_counter()
        skill = self._by_name.get(skill_name)
        if skill is None:
            return self._failure("", skill_name, "技能不在候选集中", started)

        arguments = arguments or {}
        required = set(skill.parameters.get("required", []))
        missing = sorted(required - arguments.keys())
        if missing:
            return self._failure(
                skill.id,
                skill.name,
                f"缺少参数: {', '.join(missing)}",
                started,
            )

        handler = self._handlers.get(skill_name)
        if handler is None:
            return self._failure(skill.id, skill.name, "未注册执行函数", started)

        try:
            output = handler(**arguments)
        except Exception as exc:
            return self._failure(
                skill.id,
                skill.name,
                f"{type(exc).__name__}: {exc}",
                started,
            )
        return SkillCallResult(
            skill_id=skill.id,
            skill_name=skill.name,
            success=True,
            output=str(output),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _failure(
        skill_id: str,
        skill_name: str,
        error: str,
        started: float,
    ) -> SkillCallResult:
        return SkillCallResult(
            skill_id=skill_id,
            skill_name=skill_name,
            success=False,
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
