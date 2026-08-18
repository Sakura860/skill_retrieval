"""检索器接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import RetrievalResult, Skill


class BaseRetriever(ABC):
    """Skill 检索器基类。"""

    @abstractmethod
    def index(self, skills: list[Skill]) -> None:
        """建立 Skill 索引。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """返回与查询最相关的前 k 个 Skill。"""
