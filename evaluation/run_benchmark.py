"""端到端评测入口。"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agent.agent import Agent
from core.llm import LLM
from core.schemas import Task
from data.loader import load_skills, load_tasks
from evaluation.task_metrics import evaluate_agent
from organization.hierarchical import HierarchicalOrganizer
from retrieval.evaluator import aggregate_ranking_metrics, ranking_metrics
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
    retrieval_ks: tuple[int, ...] = (1, 5, 10),
    run_id: str | None = None,
    max_steps: int = 10,
    enable_reflection: bool = True,
) -> dict:
    """在同一次检索结果上运行检索、Agent 和效率评测。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    retrieval_ks = tuple(sorted(set(retrieval_ks)))
    if not retrieval_ks or retrieval_ks[0] < 1:
        raise ValueError("retrieval_ks 必须包含正整数")

    skills = load_skills(skills_path)
    tasks = load_tasks(tasks_path)

    retriever = retriever or MultiLevelRetriever()
    organizer = organizer or HierarchicalOrganizer()
    llm = llm or LLM(provider="deepseek")
    agent = Agent(
        llm=llm,
        organizer=organizer,
        max_steps=max_steps,
        enable_reflection=enable_reflection,
        skill_handlers=skill_handlers,
    )

    retriever.index(skills)
    rankings: list[tuple[list[str], list[str]]] = []
    retrieval_top_k = max(top_k, max(retrieval_ks))

    def run_fn(task: Task) -> dict:
        retrieval = retriever.retrieve(task.instruction, top_k=retrieval_top_k)
        ranked_ids = retrieval.ranked_ids()
        gold_ids = list(task.expected_skills)
        rankings.append((ranked_ids, gold_ids))

        result = agent.run(task, retrieval.top_k(top_k))
        result["retrieved_skill_ids"] = ranked_ids
        result["retrieved_scores"] = list(retrieval.scores)
        result["retrieval_metrics"] = (
            ranking_metrics(ranked_ids, gold_ids, retrieval_ks)
            if gold_ids
            else {}
        )
        return result

    task_metrics = evaluate_agent(run_fn, tasks, success_evaluator=success_evaluator)
    retrieval_metrics = aggregate_ranking_metrics(rankings, retrieval_ks)
    timestamp = datetime.now(timezone.utc)
    resolved_run_id = run_id or timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    llm_config = {
        "provider": getattr(llm, "provider", type(llm).__name__),
        "model": getattr(llm, "model", type(llm).__name__),
        "thinking": getattr(llm, "thinking", None),
        "temperature": getattr(llm, "temperature", None),
    }

    return {
        "run_info": {
            "run_id": resolved_run_id,
            "timestamp": timestamp.isoformat(),
            "task_count": task_metrics.task_count,
            "scored_task_count": task_metrics.scored_task_count,
            "unscored_task_count": task_metrics.unscored_task_count,
        },
        "config": {
            "retriever": type(retriever).__name__,
            "retriever_backend": getattr(retriever, "backend", "native"),
            "organizer": type(organizer).__name__,
            "top_k": top_k,
            "retrieval_ks": list(retrieval_ks),
            "max_steps": max_steps,
            "enable_reflection": enable_reflection,
            "llm": llm_config,
        },
        "metrics": {
            "retrieval": retrieval_metrics,
            "agent": task_metrics.agent_metrics(),
            "efficiency": task_metrics.efficiency_metrics(),
        },
        "per_task": task_metrics.per_task,
    }
