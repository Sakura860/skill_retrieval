"""端到端评测入口。"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.agent import Agent
from core.llm import LLM
from core.schemas import RetrievalResult, Task
from data.loader import load_skills, load_tasks
from evaluation.task_metrics import evaluate_agent
from evaluation.verifiers import TaskVerifierRegistry
from execution.environment import TaskEnvironment, load_environment_fixtures
from execution.registry import SkillRegistry
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
    skill_registry: SkillRegistry | None = None,
    success_evaluator=None,
    verifier_registry: TaskVerifierRegistry | None = None,
    environment_fixtures_path: str | Path | None = None,
    task_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    top_k: int = 5,
    retrieval_ks: tuple[int, ...] = (1, 5, 10),
    run_id: str | None = None,
    max_steps: int = 10,
    enable_reflection: bool = True,
    use_task_candidate_fixtures: bool = False,
    use_task_context_budget: bool = False,
    planner_mode: str = "one_stage",
    max_argument_repairs: int = 1,
    planner_disclosure_level: str = "full",
    ensure_gold_in_retrieval: bool = False,
) -> dict:
    """在同一次检索结果上运行检索、Agent 和效率评测。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if use_task_candidate_fixtures and ensure_gold_in_retrieval:
        raise ValueError("task fixture 与 gold augmentation 不能同时启用")
    retrieval_ks = tuple(sorted(set(retrieval_ks)))
    if not retrieval_ks or retrieval_ks[0] < 1:
        raise ValueError("retrieval_ks 必须包含正整数")

    skills = load_skills(skills_path)
    tasks = load_tasks(tasks_path)
    if task_ids is not None:
        requested = set(task_ids)
        available = {task.id for task in tasks}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"未知 task_id: {', '.join(missing)}")
        tasks = [task for task in tasks if task.id in requested]

    if environment_fixtures_path is None:
        automatic_fixture_path = Path(tasks_path).parent / "environment_fixtures.json"
        if automatic_fixture_path.exists():
            environment_fixtures_path = automatic_fixture_path
    environment_fixtures = load_environment_fixtures(environment_fixtures_path)

    retriever = retriever or MultiLevelRetriever()
    organizer = organizer or HierarchicalOrganizer()
    llm = llm or LLM(provider="deepseek")
    agent = Agent(
        llm=llm,
        organizer=organizer,
        max_steps=max_steps,
        enable_reflection=enable_reflection,
        skill_handlers=skill_handlers,
        skill_registry=skill_registry,
        planner_mode=planner_mode,
        max_argument_repairs=max_argument_repairs,
        planner_disclosure_level=planner_disclosure_level,
    )

    retriever.index(skills)
    skill_by_id = {skill.id: skill for skill in skills}
    rankings: list[tuple[list[str], list[str]]] = []
    retrieval_top_k = (
        len(skills)
        if use_task_candidate_fixtures
        else max(top_k, max(retrieval_ks))
    )

    def run_fn(task: Task) -> dict:
        raw_retrieval = retriever.retrieve(
            task.instruction,
            top_k=retrieval_top_k,
        )
        raw_ranked_ids = raw_retrieval.ranked_ids()
        injected_ids: list[str] = []
        if use_task_candidate_fixtures:
            candidate_ids = list(task.metadata.get("candidate_skill_ids", []))
            if not candidate_ids:
                raise ValueError(f"任务 {task.id} 缺少 candidate_skill_ids")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"任务 {task.id} 的候选 Skill 存在重复")
            missing_ids = [item for item in candidate_ids if item not in skill_by_id]
            if missing_ids:
                raise ValueError(
                    f"任务 {task.id} 引用了未知 Skill: {', '.join(missing_ids)}"
                )
            score_by_id = dict(zip(raw_ranked_ids, raw_retrieval.scores))
            retrieval = RetrievalResult(
                query=task.instruction,
                skills=[skill_by_id[item] for item in candidate_ids],
                scores=[float(score_by_id.get(item, 0.0)) for item in candidate_ids],
            )
        elif ensure_gold_in_retrieval:
            candidate_ids = raw_ranked_ids[:top_k]
            for gold_id in task.expected_skills:
                if gold_id in candidate_ids:
                    continue
                if len(candidate_ids) >= top_k:
                    candidate_ids.pop()
                candidate_ids.append(gold_id)
                injected_ids.append(gold_id)
            score_by_id = dict(zip(raw_ranked_ids, raw_retrieval.scores))
            retrieval = RetrievalResult(
                query=task.instruction,
                skills=[skill_by_id[item] for item in candidate_ids],
                scores=[float(score_by_id.get(item, 0.0)) for item in candidate_ids],
            )
        else:
            retrieval = raw_retrieval.top_k(top_k)
        ranked_ids = retrieval.ranked_ids()
        gold_ids = list(task.expected_skills)
        metric_ranked_ids = raw_ranked_ids if ensure_gold_in_retrieval else ranked_ids
        rankings.append((metric_ranked_ids, gold_ids))
        context_budget = None
        if use_task_context_budget:
            context_budget = task.metadata.get("slice", {}).get(
                "context_budget_tokens",
                task.metadata.get("context_budget_tokens"),
            )
            if not isinstance(context_budget, int) or context_budget < 1:
                raise ValueError(f"任务 {task.id} 缺少有效的上下文预算")

        if skill_registry is None:
            result = agent.run(
                task,
                retrieval,
                context_budget_tokens=context_budget,
            )
        else:
            with TaskEnvironment(task, environment_fixtures) as environment:
                result = agent.run(
                    task,
                    retrieval,
                    environment=environment,
                    context_budget_tokens=context_budget,
                )
        result["retrieved_skill_ids"] = ranked_ids
        result["retrieved_scores"] = list(retrieval.scores)
        result["raw_bm25_skill_ids"] = raw_ranked_ids
        result["candidate_count"] = len(ranked_ids)
        result["gold_augmented_skill_ids"] = injected_ids
        result["target_gold_rank"] = task.metadata.get("slice", {}).get(
            "target_gold_rank"
        )
        result["retrieval_metrics"] = (
            ranking_metrics(metric_ranked_ids, gold_ids, retrieval_ks)
            if gold_ids
            else {}
        )
        return result

    task_metrics = evaluate_agent(
        run_fn,
        tasks,
        success_evaluator=success_evaluator,
        verifier_registry=verifier_registry,
    )
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
            "candidate_source": (
                "task_fixture"
                if use_task_candidate_fixtures
                else (
                    "retriever_top_k_gold_augmented"
                    if ensure_gold_in_retrieval
                    else "retriever_top_k"
                )
            ),
            "context_budget_source": (
                "task_slice" if use_task_context_budget else "unbounded"
            ),
            "retrieval_ks": list(retrieval_ks),
            "max_steps": max_steps,
            "enable_reflection": enable_reflection,
            "planner_mode": planner_mode,
            "max_argument_repairs": max_argument_repairs,
            "planner_disclosure_level": planner_disclosure_level,
            "task_ids": [task.id for task in tasks],
            "environment_isolated": skill_registry is not None,
            "llm": llm_config,
        },
        "metrics": {
            "retrieval": retrieval_metrics,
            "agent": task_metrics.agent_metrics(),
            "efficiency": task_metrics.efficiency_metrics(),
        },
        "per_task": task_metrics.per_task,
    }
