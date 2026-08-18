"""可训练双塔检索模型。"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.schemas import RetrievalResult, Skill
from .base import BaseRetriever


class DualEncoder(nn.Module):
    """任务塔与技能塔。"""

    def __init__(self, dim: int = 384, vocab_size: int = 10000):
        super().__init__()
        self.query_embed = nn.Embedding(vocab_size, dim)
        self.skill_embed = nn.Embedding(vocab_size, dim)
        self.dim = dim
        self.vocab_size = vocab_size

    def _bag_ids(self, text: str) -> torch.Tensor:
        ids = [
            int.from_bytes(hashlib.sha256(c.encode("utf-8")).digest()[:4], "big")
            % self.vocab_size
            for c in text[:128]
        ]
        return torch.tensor(ids or [0])

    def encode_query(self, text: str) -> torch.Tensor:
        ids = self._bag_ids(text).to(self.query_embed.weight.device)
        emb = self.query_embed(ids).mean(dim=0)
        return F.normalize(emb, dim=-1)

    def encode_skill(self, text: str) -> torch.Tensor:
        ids = self._bag_ids(text).to(self.skill_embed.weight.device)
        emb = self.skill_embed(ids).mean(dim=0)
        return F.normalize(emb, dim=-1)


class RetrievalModel(BaseRetriever):
    """DualEncoder 的检索器适配层。"""

    def __init__(self, dim: int = 384):
        self.net = DualEncoder(dim=dim)
        self._skills: list[Skill] = []
        self._skill_emb: torch.Tensor | None = None

    def forward(self, query: str, skill_text: str) -> float:
        q = self.net.encode_query(query)
        s = self.net.encode_skill(skill_text)
        return float((q * s).sum().detach())

    def index(self, skills: list[Skill]) -> None:
        self._skills = list(skills)
        if skills:
            with torch.no_grad():
                self._skill_emb = torch.stack(
                    [self.net.encode_skill(s.to_text("detailed")) for s in skills]
                )
        else:
            self._skill_emb = None

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or self._skill_emb is None:
            return RetrievalResult(query=query)
        with torch.no_grad():
            q = self.net.encode_query(query)
            sims = (self._skill_emb @ q).tolist()
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return RetrievalResult(
            query=query,
            skills=[self._skills[i] for i in order],
            scores=[sims[i] for i in order],
        )
