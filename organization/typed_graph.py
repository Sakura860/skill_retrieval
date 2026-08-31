"""带类型关系的 Skill graph 与受控 prerequisite 扩张。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.schemas import Skill


class RelationType(str, Enum):
    PREREQUISITE = "prerequisite"
    DATAFLOW = "dataflow"
    CO_USE = "co_use"
    ALTERNATIVE = "alternative"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class SkillEdge:
    source_id: str
    target_id: str
    relation_type: RelationType | str
    evidence: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_type", RelationType(self.relation_type))
        if not self.source_id or not self.target_id:
            raise ValueError("graph edge 的 source_id/target_id 不能为空")
        if self.source_id == self.target_id:
            raise ValueError("graph edge 不允许自环")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("graph edge weight 必须位于 [0, 1]")


@dataclass
class GraphExpansionResult:
    original_skill_ids: list[str]
    expanded_skill_ids: list[str]
    added_skill_ids: list[str] = field(default_factory=list)
    missing_prerequisite_ids: list[str] = field(default_factory=list)
    cyclic_skill_ids: list[str] = field(default_factory=list)
    limit_reached: bool = False


class TypedSkillGraph:
    """区分硬 prerequisite 与软关系，只有前者可自动扩张候选。"""

    def __init__(
        self,
        skills: list[Skill],
        edges: list[SkillEdge] | None = None,
    ):
        self.skills = {skill.id: skill for skill in skills}
        if len(self.skills) != len(skills):
            raise ValueError("Skill graph 中存在重复 Skill ID")
        dependency_edges = [
            SkillEdge(
                dependency,
                skill.id,
                RelationType.PREREQUISITE,
                evidence="Skill.dependencies",
            )
            for skill in skills
            for dependency in skill.dependencies
        ]
        metadata_edges = self._metadata_edges(skills)
        self.edges = self._deduplicate([*dependency_edges, *metadata_edges, *(edges or [])])

    def complete_prerequisites(
        self,
        candidate_ids: list[str],
        max_additional_skills: int = 3,
    ) -> GraphExpansionResult:
        if max_additional_skills < 0:
            raise ValueError("max_additional_skills 不能小于 0")
        unknown = [skill_id for skill_id in candidate_ids if skill_id not in self.skills]
        if unknown:
            raise ValueError(f"候选包含未知 Skill: {', '.join(unknown)}")
        original = list(dict.fromkeys(candidate_ids))
        selected = list(original)
        selected_set = set(selected)
        added: list[str] = []
        missing: list[str] = []
        limit_reached = False
        cursor = 0
        while cursor < len(selected):
            target_id = selected[cursor]
            cursor += 1
            prerequisites = sorted(
                edge.source_id
                for edge in self.edges
                if edge.relation_type == RelationType.PREREQUISITE
                and edge.target_id == target_id
            )
            for prerequisite in prerequisites:
                if prerequisite in selected_set:
                    continue
                if prerequisite not in self.skills:
                    if prerequisite not in missing:
                        missing.append(prerequisite)
                    continue
                if len(added) >= max_additional_skills:
                    limit_reached = True
                    continue
                selected.append(prerequisite)
                selected_set.add(prerequisite)
                added.append(prerequisite)

        ordered, cyclic = self._prerequisite_order(selected)
        return GraphExpansionResult(
            original_skill_ids=original,
            expanded_skill_ids=ordered,
            added_skill_ids=added,
            missing_prerequisite_ids=missing,
            cyclic_skill_ids=cyclic,
            limit_reached=limit_reached,
        )

    def infer_dataflow_edges(self, skill_ids: list[str]) -> list[SkillEdge]:
        """按返回 type 与目标参数 type 生成诊断性 dataflow 边。"""
        output: list[SkillEdge] = []
        skills = [self.skills[skill_id] for skill_id in skill_ids]
        for source in skills:
            source_type = source.returns.get("type") if source.returns else None
            if not source_type:
                continue
            for target in skills:
                if source.id == target.id:
                    continue
                properties = target.parameters.get("properties", {})
                compatible = sorted(
                    name
                    for name, schema in properties.items()
                    if self._types_compatible(source_type, schema.get("type"))
                )
                if compatible:
                    output.append(SkillEdge(
                        source.id,
                        target.id,
                        RelationType.DATAFLOW,
                        evidence=(
                            f"returns.type={source_type} -> parameters="
                            f"{','.join(compatible)}"
                        ),
                    ))
        return output

    def relation_counts(self) -> dict[str, int]:
        return {
            relation.value: sum(edge.relation_type == relation for edge in self.edges)
            for relation in RelationType
        }

    def _prerequisite_order(self, skill_ids: list[str]) -> tuple[list[str], list[str]]:
        order_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}
        selected = set(skill_ids)
        indegree = {skill_id: 0 for skill_id in skill_ids}
        adjacency = {skill_id: [] for skill_id in skill_ids}
        for edge in self.edges:
            if (
                edge.relation_type == RelationType.PREREQUISITE
                and edge.source_id in selected
                and edge.target_id in selected
            ):
                adjacency[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] += 1
        ready = sorted(
            (skill_id for skill_id, degree in indegree.items() if degree == 0),
            key=order_index.get,
        )
        ordered = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for target in sorted(adjacency[current], key=order_index.get):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=order_index.get)
        cyclic = [skill_id for skill_id in skill_ids if skill_id not in ordered]
        return [*ordered, *cyclic], cyclic

    @staticmethod
    def _metadata_edges(skills: list[Skill]) -> list[SkillEdge]:
        edges = []
        for skill in skills:
            relations = skill.metadata.get("relations", [])
            if not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                target = relation.get("target_id")
                relation_type = relation.get("type")
                if isinstance(target, str) and isinstance(relation_type, str):
                    edges.append(SkillEdge(
                        skill.id,
                        target,
                        relation_type,
                        evidence=str(relation.get("evidence", "metadata.relations")),
                        weight=float(relation.get("weight", 1.0)),
                    ))
        return edges

    @staticmethod
    def _deduplicate(edges: list[SkillEdge]) -> list[SkillEdge]:
        output = []
        seen = set()
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relation_type)
            if key not in seen:
                output.append(edge)
                seen.add(key)
        return output

    @staticmethod
    def _types_compatible(source_type: str, target_type: str | list | None) -> bool:
        targets = target_type if isinstance(target_type, list) else [target_type]
        return source_type in targets or (
            source_type == "integer" and "number" in targets
        )
