"""分层披露 Skill 上下文。"""
from __future__ import annotations

from collections import defaultdict

from core.schemas import Skill, Task
from .base import BaseOrganizer, OrganizedContext, append_whole_block


class HierarchicalOrganizer(BaseOrganizer):
    def __init__(self, detail_top_k: int = 3):
        if detail_top_k < 0:
            raise ValueError("detail_top_k 不能小于 0")
        self.detail_top_k = detail_top_k

    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        return self.organize_context(skills, task).text

    def organize_context(
        self,
        skills: list[Skill],
        task: Task | None = None,
        context_budget_tokens: int | None = None,
    ) -> OrganizedContext:
        if context_budget_tokens is not None and context_budget_tokens < 1:
            raise ValueError("context_budget_tokens 必须大于 0")
        groups: dict[str, list[Skill]] = defaultdict(list)
        for s in skills:
            groups[s.category or "other"].append(s)

        blocks: list[str] = []
        header_added = append_whole_block(
            blocks,
            "第一层：候选技能概览",
            context_budget_tokens,
        )
        exposed: list[str] = []
        truncated: list[str] = []
        budget_exhausted = not header_added
        for category, members in groups.items():
            if budget_exhausted:
                truncated.extend(s.id for s in members)
                continue
            if not append_whole_block(
                blocks,
                f"【{category}】",
                context_budget_tokens,
            ):
                truncated.extend(s.id for s in members)
                budget_exhausted = True
                continue
            for s in members:
                if budget_exhausted:
                    truncated.append(s.id)
                    continue
                block = f"- {s.to_prompt(detailed=False)}"
                if append_whole_block(blocks, block, context_budget_tokens):
                    exposed.append(s.id)
                else:
                    truncated.append(s.id)
                    budget_exhausted = True

        detailed_skills = [
            skill for skill in skills[: self.detail_top_k]
            if skill.id in exposed
        ]
        detailed: list[str] = []
        if detailed_skills:
            append_whole_block(
                blocks,
                "第二层：高排名技能详情",
                context_budget_tokens,
            )
            for index, skill in enumerate(detailed_skills, 1):
                block = f"{index}. {skill.to_prompt(detailed=True)}"
                if append_whole_block(blocks, block, context_budget_tokens):
                    detailed.append(skill.id)
                else:
                    break
        return OrganizedContext(
            text="\n".join(blocks),
            exposed_skill_ids=exposed,
            detailed_skill_ids=detailed,
            truncated_skill_ids=truncated,
            context_budget_tokens=context_budget_tokens,
        )
