"""跨模块数据结构。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Skill:
    """可检索、可调用的技能定义。"""
    id: str
    name: str
    brief_description: str
    detailed_description: str
    category: str = "other"
    parameters: dict = field(default_factory=dict)
    returns: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def description(self) -> str:
        """兼容旧代码；新代码应明确选择描述层级。"""
        return self.detailed_description or self.brief_description

    def to_text(self, level: str = "brief") -> str:
        """返回指定粒度的检索文本。"""
        if level == "brief":
            tags = " ".join(self.tags)
            return f"{self.name} {self.category} {self.brief_description} {tags}".strip()
        if level == "detailed":
            examples = " ".join(self.examples)
            return (
                f"{self.name} {self.category} {self.brief_description} "
                f"{self.detailed_description} {examples}"
            ).strip()
        raise ValueError(f"不支持的描述层级: {level}")

    def to_prompt(self, detailed: bool = False) -> str:
        """生成注入 Agent 上下文的技能说明。"""
        text = f"{self.name}: {self.brief_description}"
        if not detailed:
            return text
        details = [text, f"详细说明: {self.detailed_description}"]
        if self.parameters:
            details.append(
                f"参数Schema: {json.dumps(self.parameters, ensure_ascii=False)}"
            )
        if self.returns:
            details.append(f"返回Schema: {json.dumps(self.returns, ensure_ascii=False)}")
        if self.dependencies:
            details.append(f"依赖: {', '.join(self.dependencies)}")
        if self.examples:
            details.append(f"示例: {'; '.join(self.examples)}")
        return "\n".join(details)

    def to_schema(self) -> dict:
        """生成 Function Calling 函数定义。"""
        return {
            "name": self.name,
            "description": self.detailed_description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }


@dataclass
class Task:
    """Agent 任务及其评测标注。"""
    id: str
    instruction: str
    expected_skills: list[str] = field(default_factory=list)
    expected_skill_sequence: list[str] = field(default_factory=list)
    ground_truth: Any = None
    metadata: dict = field(default_factory=dict)

    def gold_sequence(self) -> list[str]:
        """返回顺序标注；未单独标注时沿用 expected_skills。"""
        return self.expected_skill_sequence or self.expected_skills


@dataclass
class RetrievalResult:
    """按相关性排序的技能及得分。"""
    query: str
    skills: list[Skill] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def top_k(self, k: int) -> "RetrievalResult":
        """截取前 k 项。"""
        return RetrievalResult(
            query=self.query,
            skills=self.skills[:k],
            scores=self.scores[:k],
        )

    def ranked_ids(self) -> list[str]:
        return [s.id for s in self.skills]


@dataclass
class AgentState:
    """单次 Agent 运行状态。"""
    task: Optional[Task] = None
    history: list[dict] = field(default_factory=list)
    plan: list["PlanStep"] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    finished: bool = False

    def log(self, step: str, content: str, **details: Any) -> None:
        self.history.append({"step": step, "content": content, **details})

    def to_prompt_context(self) -> str:
        """将执行轨迹转为 LLM 上下文。"""
        return "\n".join(
            f"[{item.get('step', item.get('role', 'event'))}] {item.get('content', '')}"
            for item in self.history
        )


@dataclass
class PlanStep:
    """结构化执行步骤。"""
    skill_name: str
    arguments: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class SkillCallResult:
    """一次技能调用的结构化结果。"""
    skill_id: str
    skill_name: str
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
