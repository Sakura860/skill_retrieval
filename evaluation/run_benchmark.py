"""端到端评测入口。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.agent import Agent
from core.llm import LLM
from core.schemas import Task
from data.loader import load_skills, load_tasks
from evaluation.task_metrics import evaluate_agent
from organization.hierarchical import HierarchicalOrganizer
from retrieval.multilevel import MultiLevelRetriever


def run_benchmark(
    skills_path: str,
    tasks_path: str,
    retriever=None,
    organizer=None,
    llm=None,
    skill_handlers: dict[str, Callable[..., Any]] | None = None,
    success_evaluator=None,
    top_k: int = 5,
) -> dict:
    """运行检索、组织、Agent 和任务评测。"""
    skills = load_skills(skills_path)
    tasks = load_tasks(tasks_path)

    retriever = retriever or MultiLevelRetriever()
    organizer = organizer or HierarchicalOrganizer()
    llm = llm or LLM(provider="deepseek")
    agent = Agent(llm=llm, organizer=organizer, skill_handlers=skill_handlers)

    retriever.index(skills)

    def run_fn(task: Task) -> dict:
        result = retriever.retrieve(task.instruction, top_k=top_k)
        return agent.run(task, result)

    metrics = evaluate_agent(run_fn, tasks, success_evaluator=success_evaluator)
    return {"metrics": metrics, "per_task": metrics.per_task}
