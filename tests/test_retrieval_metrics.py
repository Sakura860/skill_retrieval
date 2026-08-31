"""检索指标数值测试。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.evaluator import aggregate_ranking_metrics, ranking_metrics
from core.schemas import Skill
from retrieval.bm25 import BM25Retriever


def test_single_ranking_metrics():
    ranked = ["s1", "s2", "s3", "s4"]
    gold = ["s2", "s4"]

    metrics = ranking_metrics(ranked, gold, ks=(1, 2, 4))

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 0.5
    assert metrics["recall@4"] == 1.0
    assert metrics["mrr"] == 0.5
    expected_ndcg_at_2 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(metrics["ndcg@2"], expected_ndcg_at_2)
    assert metrics["ndcg@4"] < 1.0


def test_aggregate_ignores_tasks_without_gold_skills():
    rankings = [
        (["s1", "s2"], ["s1"]),
        (["s2", "s1"], ["s1"]),
        (["s3"], []),
    ]

    metrics = aggregate_ranking_metrics(rankings, ks=(1, 2))

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["ndcg@1"] == 0.5


def test_bm25_supports_brief_detailed_and_all_field_indexes():
    skills = [
        Skill(
            id="s1",
            name="generic_transformer",
            brief_description="转换结构化数据",
            detailed_description="处理普通对象字段。",
            parameters={"type": "object", "properties": {"data": {"type": "object"}}},
        ),
        Skill(
            id="s2",
            name="special_transformer",
            brief_description="转换结构化数据",
            detailed_description="专门处理火星气候遥测 telemetry。",
            parameters={
                "type": "object",
                "properties": {"orbital_window": {"type": "string"}},
            },
        ),
    ]

    brief = BM25Retriever(text_level="brief")
    detailed = BM25Retriever(text_level="detailed")
    all_field = BM25Retriever(text_level="all")
    for retriever in (brief, detailed, all_field):
        retriever.index(skills)

    assert brief.retrieve("telemetry", 2).ranked_ids()[0] == "s1"
    assert detailed.retrieve("telemetry", 2).ranked_ids()[0] == "s2"
    assert all_field.retrieve("orbital_window", 2).ranked_ids()[0] == "s2"


if __name__ == "__main__":
    test_single_ranking_metrics()
    test_aggregate_ignores_tasks_without_gold_skills()
    test_bm25_supports_brief_detailed_and_all_field_indexes()
    print("retrieval metrics test passed")
