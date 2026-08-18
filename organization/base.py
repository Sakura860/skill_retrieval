"""Skill 上下文组织器接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import Skill, Task


class BaseOrganizer(ABC):
    @abstractmethod
    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        """生成注入 Agent 的 Skill 上下文。"""
