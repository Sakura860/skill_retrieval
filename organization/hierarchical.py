"""分层披露 Skill 上下文。"""
from __future__ import annotations

from collections import defaultdict

from core.schemas import Skill, Task
from .base import BaseOrganizer


class HierarchicalOrganizer(BaseOrganizer):
    def __init__(self, detail_top_k: int = 3):
        if detail_top_k < 0:
            raise ValueError("detail_top_k 不能小于 0")
        self.detail_top_k = detail_top_k

    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        groups: dict[str, list[Skill]] = defaultdict(list)
        for s in skills:
            groups[s.category or "other"].append(s)

        lines = ["第一层：候选技能概览"]
        for category, members in groups.items():
            lines.append(f"\n【{category}】")
            for s in members:
                lines.append(f"- {s.to_prompt(detailed=False)}")

        detailed_skills = skills[: self.detail_top_k]
        if detailed_skills:
            lines.append("\n第二层：高排名技能详情")
            for index, skill in enumerate(detailed_skills, 1):
                lines.append(f"\n{index}. {skill.to_prompt(detailed=True)}")
        return "\n".join(lines)
