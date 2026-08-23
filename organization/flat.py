"""扁平 Skill 上下文。"""
from __future__ import annotations

from core.schemas import Skill, Task
from .base import BaseOrganizer, OrganizedContext, append_whole_block


class FlatOrganizer(BaseOrganizer):
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
        blocks: list[str] = []
        header_added = append_whole_block(
            blocks,
            "可用技能：",
            context_budget_tokens,
        )
        exposed: list[str] = []
        truncated: list[str] = []
        for index, skill in enumerate(skills, 1):
            if not header_added:
                truncated.append(skill.id)
                continue
            block = f"{index}. {skill.to_prompt(detailed=True)}"
            if append_whole_block(blocks, block, context_budget_tokens):
                exposed.append(skill.id)
            else:
                truncated.extend(item.id for item in skills[index - 1:])
                break
        return OrganizedContext(
            text="\n".join(blocks),
            exposed_skill_ids=exposed,
            detailed_skill_ids=list(exposed),
            truncated_skill_ids=truncated,
            context_budget_tokens=context_budget_tokens,
        )
