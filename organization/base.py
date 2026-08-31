"""Skill 上下文组织器接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.schemas import Skill, Task
from core.token_utils import estimate_tokens


@dataclass(frozen=True)
class OrganizedContext:
    """组织后的上下文及可审计披露信息。"""

    text: str
    exposed_skill_ids: list[str] = field(default_factory=list)
    detailed_skill_ids: list[str] = field(default_factory=list)
    truncated_skill_ids: list[str] = field(default_factory=list)
    added_skill_ids: list[str] = field(default_factory=list)
    graph_issues: list[str] = field(default_factory=list)
    resolved_skills: list[Skill] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )
    context_budget_tokens: int | None = None

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)


def append_whole_block(
    accepted: list[str],
    block: str,
    context_budget_tokens: int | None,
) -> bool:
    """仅在整个块可放入预算时追加，绝不截断 Skill 定义。"""
    candidate = "\n".join([*accepted, block])
    if (
        context_budget_tokens is not None
        and estimate_tokens(candidate) > context_budget_tokens
    ):
        return False
    accepted.append(block)
    return True


class BaseOrganizer(ABC):
    @abstractmethod
    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        """生成注入 Agent 的 Skill 上下文。"""

    def organize_context(
        self,
        skills: list[Skill],
        task: Task | None = None,
        context_budget_tokens: int | None = None,
    ) -> OrganizedContext:
        """兼容旧组织器；新组织器应覆盖此方法并返回披露证据。"""
        if context_budget_tokens is not None and context_budget_tokens < 1:
            raise ValueError("context_budget_tokens 必须大于 0")
        text = self.organize(skills, task)
        if (
            context_budget_tokens is not None
            and estimate_tokens(text) > context_budget_tokens
        ):
            raise ValueError("旧组织器不支持在给定预算下做完整块裁剪")
        skill_ids = [skill.id for skill in skills]
        return OrganizedContext(
            text=text,
            exposed_skill_ids=skill_ids,
            detailed_skill_ids=skill_ids,
            context_budget_tokens=context_budget_tokens,
        )
