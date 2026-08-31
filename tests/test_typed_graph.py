"""typed Skill graph、prerequisite completion 与 dataflow 诊断测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.schemas import Skill
from core.token_utils import estimate_tokens
from organization.graph import GraphOrganizer
from organization.typed_graph import RelationType, SkillEdge, TypedSkillGraph


def _skill(
    skill_id: str,
    dependencies: list[str] | None = None,
    returns_type: str = "string",
    input_type: str = "string",
) -> Skill:
    return Skill(
        id=skill_id,
        name=f"skill_{skill_id}",
        brief_description=f"brief {skill_id}",
        detailed_description=f"details {skill_id}",
        parameters={
            "type": "object",
            "properties": {"value": {"type": input_type}},
            "required": ["value"],
        },
        returns={"type": returns_type},
        dependencies=dependencies or [],
    )


def test_only_prerequisites_expand_candidates_and_order_before_dependents():
    prepare = _skill("prepare", returns_type="object")
    execute = _skill("execute", ["prepare"], input_type="object")
    alternative = _skill("alternative")
    graph = TypedSkillGraph(
        [prepare, execute, alternative],
        [
            SkillEdge(
                "alternative",
                "execute",
                RelationType.ALTERNATIVE,
                evidence="same user intent",
            ),
            SkillEdge(
                "alternative",
                "prepare",
                RelationType.CONFLICT,
                evidence="incompatible state boundary",
            ),
        ],
    )

    result = graph.complete_prerequisites(["execute"], max_additional_skills=1)

    assert result.expanded_skill_ids == ["prepare", "execute"]
    assert result.added_skill_ids == ["prepare"]
    assert "alternative" not in result.expanded_skill_ids
    assert graph.relation_counts()["prerequisite"] == 1
    assert graph.relation_counts()["alternative"] == 1
    assert graph.relation_counts()["conflict"] == 1


def test_expansion_limit_and_missing_prerequisites_are_auditable():
    root = _skill("root", ["missing_external"])
    middle = _skill("middle", ["root"])
    target = _skill("target", ["middle"])
    graph = TypedSkillGraph([root, middle, target])

    limited = graph.complete_prerequisites(["target"], max_additional_skills=1)
    assert limited.added_skill_ids == ["middle"]
    assert limited.limit_reached is True

    complete = graph.complete_prerequisites(["target"], max_additional_skills=3)
    assert complete.expanded_skill_ids == ["root", "middle", "target"]
    assert complete.missing_prerequisite_ids == ["missing_external"]


def test_schema_compatibility_infers_diagnostic_dataflow_edges():
    source = _skill("source", returns_type="object")
    compatible = _skill("compatible", input_type="object")
    incompatible = _skill("incompatible", input_type="array")
    graph = TypedSkillGraph([source, compatible, incompatible])

    edges = graph.infer_dataflow_edges(["source", "compatible", "incompatible"])
    pairs = {(edge.source_id, edge.target_id) for edge in edges}

    assert ("source", "compatible") in pairs
    assert ("source", "incompatible") not in pairs
    assert all(edge.relation_type == RelationType.DATAFLOW for edge in edges)


def test_typed_prerequisite_cycles_are_reported_without_hanging():
    first = _skill("first", ["second"])
    second = _skill("second", ["first"])
    result = TypedSkillGraph([first, second]).complete_prerequisites(
        ["first"],
        max_additional_skills=1,
    )

    assert result.added_skill_ids == ["second"]
    assert result.cyclic_skill_ids == ["first", "second"]


def test_graph_organizer_exposes_and_returns_added_prerequisites():
    prepare = _skill("prepare", returns_type="object")
    execute = _skill("execute", ["prepare"], input_type="object")
    organized = GraphOrganizer(
        catalog=[prepare, execute],
        max_additional_skills=1,
    ).organize_context([execute])

    assert organized.added_skill_ids == ["prepare"]
    assert organized.exposed_skill_ids == ["prepare", "execute"]
    assert [skill.id for skill in organized.resolved_skills] == ["prepare", "execute"]
    assert organized.text.index("skill_prepare") < organized.text.index("skill_execute")

    prerequisite_only = GraphOrganizer().organize_context([prepare])
    budgeted = GraphOrganizer(
        catalog=[prepare, execute],
        max_additional_skills=1,
    ).organize_context(
        [execute],
        context_budget_tokens=prerequisite_only.token_count,
    )
    assert estimate_tokens(budgeted.text) <= prerequisite_only.token_count
    assert budgeted.exposed_skill_ids == ["prepare"]
    assert budgeted.truncated_skill_ids == ["execute"]


if __name__ == "__main__":
    test_only_prerequisites_expand_candidates_and_order_before_dependents()
    test_expansion_limit_and_missing_prerequisites_are_auditable()
    test_schema_compatibility_infers_diagnostic_dataflow_edges()
    test_typed_prerequisite_cycles_are_reported_without_hanging()
    test_graph_organizer_exposes_and_returns_added_prerequisites()
    print("typed graph tests passed")
