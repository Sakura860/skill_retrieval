"""集中管理 Skill handler 与执行权限。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .environment import TaskEnvironment

EnvironmentHandler = Callable[..., Any]


@dataclass(frozen=True)
class RegisteredSkill:
    name: str
    handler: EnvironmentHandler
    permissions: frozenset[str]


class SkillRegistry:
    """显式注册 handler，并在绑定 Task 环境时检查权限。"""

    def __init__(self):
        self._skills: dict[str, RegisteredSkill] = {}

    def register(
        self,
        name: str,
        handler: EnvironmentHandler,
        permissions: set[str] | frozenset[str] | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if not name:
            raise ValueError("Skill 名称不能为空")
        if name in self._skills and not replace:
            raise ValueError(f"Skill handler 已注册: {name}")
        self._skills[name] = RegisteredSkill(
            name=name,
            handler=handler,
            permissions=frozenset(permissions or set()),
        )

    def handlers_for(self, environment: TaskEnvironment) -> dict[str, Callable]:
        handlers: dict[str, Callable] = {}
        for name, registered in self._skills.items():
            handlers[name] = self._bind(registered, environment)
        return handlers

    def names(self) -> set[str]:
        return set(self._skills)

    @staticmethod
    def _bind(
        registered: RegisteredSkill,
        environment: TaskEnvironment,
    ) -> Callable:
        @wraps(registered.handler)
        def bound(**arguments: Any) -> Any:
            if not environment.has_permissions(registered.permissions):
                missing = sorted(
                    registered.permissions - environment.allowed_permissions
                )
                raise PermissionError(f"Task 环境缺少权限: {', '.join(missing)}")
            return registered.handler(environment, **arguments)

        return bound
