"""粗排与细排结合的两阶段检索。"""
from __future__ import annotations

from core.schemas import RetrievalResult, Skill

from .base import BaseRetriever
from .bm25 import BM25Retriever


class MultiLevelRetriever(BaseRetriever):
    """先检索粗略描述，再用详细描述重排候选。"""

    def __init__(self, coarse_k: int = 20, coarse_weight: float = 0.4):
        if coarse_k < 1:
            raise ValueError("coarse_k 必须大于 0")
        if not 0.0 <= coarse_weight <= 1.0:
            raise ValueError("coarse_weight 必须位于 [0, 1]")
        self.coarse_k = coarse_k
        self.coarse_weight = coarse_weight
        self._skills: list[Skill] = []
        self._coarse = BM25Retriever(text_level="brief")
        self._fine = BM25Retriever(text_level="detailed")

    def index(self, skills: list[Skill]) -> None:
        self._skills = list(skills)
        self._coarse.index(self._skills)
        self._fine.index(self._skills)

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        maximum = max(scores, default=0.0)
        if maximum <= 0:
            return [0.0 for _ in scores]
        return [score / maximum for score in scores]

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or not self._skills:
            return RetrievalResult(query=query)

        candidate_k = min(max(top_k, self.coarse_k), len(self._skills))
        coarse = self._coarse.retrieve(query, candidate_k)
        fine = self._fine.retrieve(query, len(self._skills))

        coarse_scores = dict(zip(coarse.ranked_ids(), self._normalize(coarse.scores)))
        fine_scores = dict(zip(fine.ranked_ids(), self._normalize(fine.scores)))
        combined = {
            skill.id: self.coarse_weight * coarse_scores.get(skill.id, 0.0)
            + (1.0 - self.coarse_weight) * fine_scores.get(skill.id, 0.0)
            for skill in coarse.skills
        }
        ranked = sorted(coarse.skills, key=lambda skill: combined[skill.id], reverse=True)[:top_k]
        return RetrievalResult(
            query=query,
            skills=ranked,
            scores=[combined[skill.id] for skill in ranked],
        )
