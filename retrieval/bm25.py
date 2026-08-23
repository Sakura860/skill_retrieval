"""无第三方依赖的 BM25 检索。"""
from __future__ import annotations

import math
from collections import Counter

from core.schemas import RetrievalResult, Skill
from .base import BaseRetriever
from .tokenize import tokenize


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75, text_level: str = "brief"):
        self.k1, self.b = k1, b
        self.text_level = text_level
        self._docs: list[Skill] = []
        self._tokens: list[list[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._N: int = 0

    def index(self, skills: list[Skill]) -> None:
        self._docs = list(skills)
        self._tokens = [tokenize(s.to_text(self.text_level)) for s in skills]
        self._N = len(skills)
        self._df = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        self._avgdl = sum(len(t) for t in self._tokens) / max(self._N, 1)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or not self._docs:
            return RetrievalResult(query=query)
        q_tokens = tokenize(query)
        scores: list[float] = []
        for doc_tokens in self._tokens:
            scores.append(self._score(q_tokens, doc_tokens))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return RetrievalResult(
            query=query,
            skills=[self._docs[i] for i in order],
            scores=[scores[i] for i in order],
        )

    def _score(self, q_tokens: list[str], doc_tokens: list[str]) -> float:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for q in sorted(set(q_tokens)):
            if q not in tf:
                continue
            n = self._df.get(q, 0)
            idf = math.log(1 + (self._N - n + 0.5) / (n + 0.5))
            denom = tf[q] + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
            score += idf * tf[q] * (self.k1 + 1) / denom
        return score
