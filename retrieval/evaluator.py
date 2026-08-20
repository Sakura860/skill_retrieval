"""检索指标：Recall@k、MRR、NDCG@k。"""
from __future__ import annotations

import math

from core.schemas import Skill, Task
from .base import BaseRetriever


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    top = ranked_ids[:k]
    return sum(1 for g in gold_ids if g in top) / max(len(gold_ids), 1)


def mrr(ranked_ids: list[str], gold_ids: list[str]) -> float:
    for i, sid in enumerate(ranked_ids):
        if sid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(ranked_ids[:k]) if sid in gold_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(
    ranked_ids: list[str],
    gold_ids: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """计算单个任务的检索指标。"""
    metrics = {"mrr": mrr(ranked_ids, gold_ids)}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, gold_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, gold_ids, k)
    return metrics


def aggregate_ranking_metrics(
    rankings: list[tuple[list[str], list[str]]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """对有检索标注的任务做宏平均。"""
    valid = [(ranked, gold) for ranked, gold in rankings if gold]
    if not valid:
        return {}

    aggregate: dict[str, float] = {}
    for ranked_ids, gold_ids in valid:
        for name, value in ranking_metrics(ranked_ids, gold_ids, ks).items():
            aggregate[name] = aggregate.get(name, 0.0) + value
    return {name: value / len(valid) for name, value in aggregate.items()}


def evaluate_retrieval(
    retriever: BaseRetriever,
    tasks: list[Task],
    skills: list[Skill],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """计算任务集上的平均检索指标。"""
    retriever.index(skills)
    rankings: list[tuple[list[str], list[str]]] = []
    for task in tasks:
        if not task.expected_skills:
            continue
        res = retriever.retrieve(task.instruction, top_k=max(ks))
        rankings.append((res.ranked_ids(), list(task.expected_skills)))
    return aggregate_ranking_metrics(rankings, ks)
