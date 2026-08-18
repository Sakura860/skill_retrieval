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


def evaluate_retrieval(
    retriever: BaseRetriever,
    tasks: list[Task],
    skills: list[Skill],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """计算任务集上的平均检索指标。"""
    retriever.index(skills)
    agg: dict[str, float] = {}
    n = 0
    for task in tasks:
        if not task.expected_skills:
            continue
        res = retriever.retrieve(task.instruction, top_k=max(ks))
        ranked = res.ranked_ids()
        gold = task.expected_skills
        n += 1
        agg.setdefault("mrr", 0.0)
        agg["mrr"] += mrr(ranked, gold)
        for k in ks:
            agg.setdefault(f"recall@{k}", 0.0)
            agg.setdefault(f"ndcg@{k}", 0.0)
            agg[f"recall@{k}"] += recall_at_k(ranked, gold, k)
            agg[f"ndcg@{k}"] += ndcg_at_k(ranked, gold, k)

    if n == 0:
        return {}
    return {k: v / n for k, v in agg.items()}
