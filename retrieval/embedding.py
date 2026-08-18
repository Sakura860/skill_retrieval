"""SentenceTransformer 稠密检索，失败时使用 Hash 向量。"""
from __future__ import annotations

import hashlib

import numpy as np

from core.schemas import RetrievalResult, Skill
from .base import BaseRetriever
from .tokenize import tokenize


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        text_level: str = "brief",
        allow_hash_fallback: bool = True,
    ):
        self.model_name = model_name
        self.text_level = text_level
        self.allow_hash_fallback = allow_hash_fallback
        self.backend = "uninitialized"
        self.fallback_reason = ""
        self._model = None
        self._skills: list[Skill] = []
        self._emb: np.ndarray | None = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception as exc:
            if not self.allow_hash_fallback:
                raise
            self._model = False
            self.backend = "hash"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        if self._model is not False:
            return self._model.encode(texts, normalize_embeddings=True)
        vecs = []
        for t in texts:
            v = np.zeros(128, dtype=np.float32)
            for token in tokenize(t):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                v[h % 128] += 1.0
            vecs.append(v)
        if not vecs:
            return np.empty((0, 128), dtype=np.float32)
        arr = np.stack(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / norms

    def index(self, skills: list[Skill]) -> None:
        self._skills = list(skills)
        self._emb = self._encode([s.to_text(self.text_level) for s in skills])

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or self._emb is None or not self._skills:
            return RetrievalResult(query=query)
        q = self._encode([query])[0]
        sims = (self._emb @ q).tolist()
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return RetrievalResult(
            query=query,
            skills=[self._skills[i] for i in order],
            scores=[sims[i] for i in order],
        )
