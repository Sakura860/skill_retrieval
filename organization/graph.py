"""按依赖顺序组织 Skill。"""
from __future__ import annotations

from collections import defaultdict, deque

from core.schemas import Skill, Task
from .base import BaseOrganizer


class GraphOrganizer(BaseOrganizer):
    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
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

        lines = ["可用技能（依赖顺序）："]
        for sid in order:
            s = by_id[sid]
            lines.append(f"\n- {s.to_prompt(detailed=True)}")
        if missing_dependencies:
            lines.append(f"\n警告：候选集中缺少依赖：{', '.join(missing_dependencies)}")
        if cyclic:
            lines.append(f"\n警告：检测到依赖环：{', '.join(cyclic)}")
        return "\n".join(lines)
