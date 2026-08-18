"""Agent 状态管理。"""
from __future__ import annotations

from typing import Any

from core.schemas import AgentState, Task


class AgentStateManager:
    def __init__(self, task: Task):
        self.state = AgentState(task=task)

    def log(self, step: str, content: str, **details: Any) -> None:
        self.state.log(step, content, **details)

    @property
    def finished(self) -> bool:
        return self.state.finished
