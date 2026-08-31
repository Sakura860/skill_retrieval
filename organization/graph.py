"""按依赖顺序组织 Skill。"""
from __future__ import annotations

from collections import defaultdict, deque

from core.schemas import Skill, Task
from .base import BaseOrganizer, OrganizedContext, append_whole_block
from .typed_graph import SkillEdge, TypedSkillGraph


class GraphOrganizer(BaseOrganizer):
    def __init__(
        self,
        catalog: list[Skill] | None = None,
        edges: list[SkillEdge] | None = None,
        max_additional_skills: int = 0,
    ):
        if max_additional_skills < 0:
            raise ValueError("max_additional_skills 不能小于 0")
        self.catalog = list(catalog or [])
        self.edges = list(edges or [])
        self.max_additional_skills = max_additional_skills

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
        added_skill_ids: list[str] = []
        graph_issues: list[str] = []
        if self.catalog:
            task_edges = []
            if task is not None:
                for item in task.metadata.get("graph_edges", []):
                    if isinstance(item, dict):
                        task_edges.append(SkillEdge(
                            source_id=item["source_id"],
                            target_id=item["target_id"],
                            relation_type=item.get("relation_type", item.get("type")),
                            evidence=str(item.get("evidence", "task.metadata.graph_edges")),
                            weight=float(item.get("weight", 1.0)),
                        ))
            graph = TypedSkillGraph(self.catalog, [*self.edges, *task_edges])
            expansion = graph.complete_prerequisites(
                [skill.id for skill in skills],
                self.max_additional_skills,
            )
            skills = [graph.skills[skill_id] for skill_id in expansion.expanded_skill_ids]
            added_skill_ids = expansion.added_skill_ids
            if expansion.missing_prerequisite_ids:
                graph_issues.append(
                    "missing_prerequisites:"
                    + ",".join(expansion.missing_prerequisite_ids)
                )
            if expansion.cyclic_skill_ids:
                graph_issues.append("cycle:" + ",".join(expansion.cyclic_skill_ids))
            if expansion.limit_reached:
                graph_issues.append("expansion_limit_reached")
        by_id = {s.id: s for s in skills}
        missing_dependencies = sorted({
            dependency
            for skill in skills
            for dependency in skill.dependencies
            if dependency not in by_id
        })
        missing_issue = "missing_prerequisites:" + ",".join(missing_dependencies)
        if missing_dependencies and missing_issue not in graph_issues:
            graph_issues.append(missing_issue)
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
        cycle_issue = "cycle:" + ",".join(cyclic)
        if cyclic and cycle_issue not in graph_issues:
            graph_issues.append(cycle_issue)
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
        if added_skill_ids:
            append_whole_block(
                blocks,
                f"图补充的前置技能：{', '.join(added_skill_ids)}",
                context_budget_tokens,
            )
        return OrganizedContext(
            text="\n".join(blocks),
            exposed_skill_ids=exposed,
            detailed_skill_ids=list(exposed),
            truncated_skill_ids=truncated,
            added_skill_ids=added_skill_ids,
            graph_issues=graph_issues,
            resolved_skills=list(skills),
            context_budget_tokens=context_budget_tokens,
        )
