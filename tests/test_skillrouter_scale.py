"""Offline tests for the frozen large-pool scaling runner."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.schemas import Skill  # noqa: E402
from experiments.run_skillrouter_scale import (  # noqa: E402
    CompactBM25Index,
    aggregate_metrics,
    bm25_text,
    metrics,
    ordered_pool,
)
from retrieval.bm25 import BM25Retriever  # noqa: E402


def records() -> list[dict]:
    return [
        {
            "skill_id": "s-alpha",
            "name": "alpha calculator",
            "description": "compute alpha totals",
            "body": "sums a numeric list without sorting",
        },
        {
            "skill_id": "s-beta",
            "name": "beta formatter",
            "description": "format beta text",
            "body": "preserves whitespace exactly",
        },
        {
            "skill_id": "s-gamma",
            "name": "gamma database",
            "description": "query gamma records",
            "body": "read-only structured query",
        },
    ]


def test_compact_index_matches_reference_bm25() -> None:
    source = records()
    compact = CompactBM25Index()
    compact.build(
        source,
        lambda item: bm25_text(item, "all"),
        checkpoint_sizes=[3],
        seed="seed",
        tier="easy",
    )
    reference_skills = [
        Skill(
            id=item["skill_id"],
            name=item["name"],
            brief_description=item["description"],
            detailed_description=item["body"],
        )
        for item in source
    ]
    reference = BM25Retriever(text_level="detailed")
    reference.index(reference_skills)
    expected = reference.retrieve("numeric list totals", top_k=3).ranked_ids()
    actual = compact.retrieve("numeric list totals", active_size=3, top_k=3)
    assert actual[0] == expected[0] == "s-alpha"
    assert compact.checkpoints[3]["total_document_tokens"] > 0


def test_scale_membership_keeps_all_graded_items() -> None:
    source = records()
    ordered, required_count = ordered_pool(
        source,
        {"s-beta", "s-gamma"},
        seed="seed",
        tier="hard",
    )
    assert required_count == 2
    assert {item["skill_id"] for item in ordered[:2]} == {"s-beta", "s-gamma"}


def test_official_metric_shapes() -> None:
    row = metrics(
        ["a", "b", "c"],
        {"a", "c"},
        {"a": 3.0, "c": 3.0, "b": 1.0},
    )
    assert row["Hit@1"] == 1.0
    assert row["Recall@10"] == 1.0
    assert row["FullCoverage@3"] == 1.0
    aggregate = aggregate_metrics([row, row])
    assert aggregate["count"] == 2
    assert aggregate["MRR@10"] == 1.0


def main() -> None:
    test_compact_index_matches_reference_bm25()
    test_scale_membership_keeps_all_graded_items()
    test_official_metric_shapes()
    print("SkillRouter scale tests passed")


if __name__ == "__main__":
    main()
