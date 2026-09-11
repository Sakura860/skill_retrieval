"""可训练双塔检索模型。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_ARCHITECTURE = "skill-agent-dual-encoder"
TOKENIZATION_VERSION = "sha256-char-utf8-first128-v1"


class RetrievalModel(BaseRetriever):
    """DualEncoder 的检索器适配层。"""

    def __init__(
        self,
        dim: int = 384,
        vocab_size: int = 10000,
        text_level: str = "detailed",
        seed: int = 0,
    ):
        if dim <= 0:
            raise ValueError("dim 必须大于 0")
        if vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0")
        if text_level not in {"brief", "detailed", "all", "body"}:
            raise ValueError("text_level 必须是 brief、detailed、all 或 body")
        self.dim = dim
        self.vocab_size = vocab_size
        self.text_level = text_level
        self.seed = seed
        # 初始化可复现，但不污染调用方的全局 RNG 状态。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = DualEncoder(dim=dim, vocab_size=vocab_size)
        self._skills: list[Skill] = []
        self._skill_emb: torch.Tensor | None = None
        self.training_history: list[float] = []
        self.checkpoint_metadata: dict[str, Any] = {}

    def forward(self, query: str, skill_text: str) -> float:
        q = self.net.encode_query(query)
        s = self.net.encode_skill(skill_text)
        return float((q * s).sum().detach())

    def index(self, skills: list[Skill]) -> None:
        self._skills = list(skills)
        if skills:
            with torch.no_grad():
                self._skill_emb = torch.stack(
                    [self.net.encode_skill(s.to_text(self.text_level)) for s in skills]
                )
        else:
            self._skill_emb = None

    def invalidate_index(self) -> None:
        """训练或加载权重后清除可能过期的 Skill 向量。"""
        self._skills = []
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

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """保存可移植的 CPU 权重与构造参数，不持久化临时索引。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in self.net.state_dict().items()
        }
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": CHECKPOINT_ARCHITECTURE,
            "tokenization_version": TOKENIZATION_VERSION,
            "config": {
                "dim": self.dim,
                "vocab_size": self.vocab_size,
                "text_level": self.text_level,
                "seed": self.seed,
            },
            "state_dict": state_dict,
            "training_history": list(self.training_history),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "RetrievalModel":
        """校验格式并加载模型；调用方需用自己的 Skill 集重新建索引。"""
        source = Path(path)
        try:
            payload = torch.load(
                source,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:  # 兼容较早的 torch 2.x。
            payload = torch.load(source, map_location=map_location)
        if not isinstance(payload, dict):
            raise ValueError("双塔 checkpoint 顶层必须是字典")
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("不支持的双塔 checkpoint 格式版本")
        if payload.get("architecture") != CHECKPOINT_ARCHITECTURE:
            raise ValueError("checkpoint 架构与 RetrievalModel 不匹配")
        if payload.get("tokenization_version") != TOKENIZATION_VERSION:
            raise ValueError("checkpoint 的文本哈希版本与当前实现不匹配")
        config = payload.get("config")
        state_dict = payload.get("state_dict")
        if not isinstance(config, dict) or not isinstance(state_dict, dict):
            raise ValueError("双塔 checkpoint 缺少 config 或 state_dict")
        required = {"dim", "vocab_size", "text_level", "seed"}
        if not required.issubset(config):
            raise ValueError("双塔 checkpoint 的构造参数不完整")

        model = cls(
            dim=int(config["dim"]),
            vocab_size=int(config["vocab_size"]),
            text_level=str(config["text_level"]),
            seed=int(config["seed"]),
        )
        model.net.load_state_dict(state_dict, strict=True)
        model.net.to(map_location)
        history = payload.get("training_history", [])
        if not isinstance(history, list) or not all(
            isinstance(value, (int, float)) for value in history
        ):
            raise ValueError("双塔 checkpoint 的 training_history 无效")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("双塔 checkpoint 的 metadata 无效")
        model.training_history = [float(value) for value in history]
        model.checkpoint_metadata = dict(metadata)
        model.invalidate_index()
        return model
