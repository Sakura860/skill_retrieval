"""扁平 Skill 上下文。"""
from __future__ import annotations

from core.schemas import Skill, Task
from .base import BaseOrganizer


class FlatOrganizer(BaseOrganizer):
    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        lines = ["可用技能："]
        for i, s in enumerate(skills, 1):
            lines.append(f"\n{i}. {s.to_prompt(detailed=True)}")
        return "\n".join(lines)
