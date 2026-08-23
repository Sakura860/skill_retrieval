"""按依赖顺序组织 Skill。"""
from __future__ import annotations

from collections import defaultdict, deque

from core.schemas import Skill, Task
from .base import BaseOrganizer, OrganizedContext, append_whole_block


class GraphOrganizer(BaseOrganizer):
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
        by_id = {s.id: s for s in skills}
        missing_dependencies = sorted({
            dependency
            for skill in skills
            for dependency in skill.dependencies
            if dependency not in by_id
        })
        indeg = {s.id: 0 for s in skills}
        edges: dict[str, list[str]] = defaultdict(list)
        for s in skills:
            for dep in s.dependencies:
                if dep in by_id:
                    edges[dep].append(s.id)
                    indeg[s.id] += 1

        q = deque(sid for sid, d in indeg.items() if d == 0)
        order: list[str] = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in edges[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)

        cyclic = [sid for sid in indeg if sid not in order]
        order += cyclic

        blocks: list[str] = []
        header_added = append_whole_block(
            blocks,
            "可用技能（依赖顺序）：",
            context_budget_tokens,
        )
        exposed: list[str] = []
        truncated: list[str] = []
        for index, sid in enumerate(order):
            if not header_added:
                truncated.append(sid)
                continue
            s = by_id[sid]
            block = f"- {s.to_prompt(detailed=True)}"
            if append_whole_block(blocks, block, context_budget_tokens):
                exposed.append(sid)
            else:
                truncated.extend(order[index:])
                break
        if missing_dependencies:
            append_whole_block(
                blocks,
                f"警告：候选集中缺少依赖：{', '.join(missing_dependencies)}",
                context_budget_tokens,
            )
        if cyclic:
            append_whole_block(
                blocks,
                f"警告：检测到依赖环：{', '.join(cyclic)}",
                context_budget_tokens,
            )
        return OrganizedContext(
            text="\n".join(blocks),
            exposed_skill_ids=exposed,
            detailed_skill_ids=list(exposed),
            truncated_skill_ids=truncated,
            context_budget_tokens=context_budget_tokens,
        )
