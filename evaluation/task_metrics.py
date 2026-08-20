"""Agent 任务级评测。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass
class TaskMetrics:
    """端到端指标汇总。"""

    skill_selection_f1: float = 0.0
    sequence_accuracy: float = 0.0
    task_success_rate: float | None = None
    avg_skill_calls: float = 0.0
    redundant_call_rate: float = 0.0
    skill_context_tokens: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    avg_execution_steps: float = 0.0
    avg_execution_time_ms: float = 0.0
    task_count: int = 0
    scored_task_count: int = 0
    unscored_task_count: int = 0
    per_task: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "skill_selection_f1": self.skill_selection_f1,
            "sequence_accuracy": self.sequence_accuracy,
            "task_success_rate": self.task_success_rate,
            "avg_skill_calls": self.avg_skill_calls,
            "redundant_call_rate": self.redundant_call_rate,
            "skill_context_tokens": self.skill_context_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_execution_steps": self.avg_execution_steps,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "task_count": self.task_count,
            "scored_task_count": self.scored_task_count,
            "unscored_task_count": self.unscored_task_count,
        }

    def agent_metrics(self) -> dict[str, float | None]:
        return {
            "skill_selection_f1": self.skill_selection_f1,
            "sequence_accuracy": self.sequence_accuracy,
            "task_success_rate": self.task_success_rate,
        }

    def efficiency_metrics(self) -> dict[str, float | int]:
        avg_tokens = self.total_tokens / self.task_count if self.task_count else 0.0
        return {
            "avg_skill_context_tokens": self.skill_context_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_total_tokens_per_task": avg_tokens,
            "avg_skill_calls": self.avg_skill_calls,
            "redundant_call_rate": self.redundant_call_rate,
            "avg_execution_steps": self.avg_execution_steps,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }


def _selection_f1(selected: list[str], expected: list[str]) -> float:
    predicted_set = set(selected)
    expected_set = set(expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    true_positive = len(predicted_set & expected_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(expected_set)
    return 2 * precision * recall / (precision + recall) if true_positive else 0.0


def _redundant_calls(selected: list[str], expected_sequence: list[str]) -> int:
    remaining = Counter(expected_sequence)
    redundant = 0
    for skill_id in selected:
        if remaining[skill_id] > 0:
            remaining[skill_id] -= 1
        else:
            redundant += 1
    return redundant


def _default_success(task: Any, result: dict) -> bool | None:
    ground_truth = getattr(task, "ground_truth", None)
    if ground_truth is None:
        return None
    if not result.get("success", False):
        return False
    answer = str(result.get("answer", "")).strip().casefold()
    expected = str(ground_truth).strip().casefold()
    try:
        return Decimal(answer) == Decimal(expected)
    except InvalidOperation:
        pass
    return answer == expected


def evaluate_agent(
    run_fn: Callable[[Any], dict],
    tasks: list[Any],
    success_evaluator: Callable[[Any, dict], bool | None] | None = None,
) -> TaskMetrics:
    """执行任务并汇总 Agent、任务成功与效率指标。"""
    metrics = TaskMetrics(task_count=len(tasks))
    if not tasks:
        return metrics

    judge = success_evaluator or _default_success
    total_redundant = 0
    total_calls = 0
    successful_tasks = 0

    for task in tasks:
        result = run_fn(task)
        selected = list(result.get("selected_skill_ids", []))
        expected = list(getattr(task, "expected_skills", []))
        expected_sequence = list(task.gold_sequence())
        calls = int(result.get("skill_calls", len(selected)))
        redundant = _redundant_calls(selected, expected_sequence)
        judged = judge(task, result)
        scored = judged is not None
        task_success = bool(judged) if scored else None
        if scored:
            metrics.scored_task_count += 1
            successful_tasks += int(task_success)
        else:
            metrics.unscored_task_count += 1
        selection_f1 = _selection_f1(selected, expected)
        sequence_correct = selected == expected_sequence
        context_tokens = int(result.get("skill_context_tokens", 0))
        token_usage = dict(result.get("token_usage", {}))
        prompt_tokens = int(token_usage.get("prompt_tokens", 0))
        completion_tokens = int(token_usage.get("completion_tokens", 0))
        task_tokens = int(token_usage.get("total_tokens", 0))
        execution_steps = int(result.get("execution_steps", 0))
        execution_time_ms = float(result.get("execution_time_ms", 0.0))
        execution_success = bool(result.get("success", False))
        if success_evaluator is not None:
            verifier_type = getattr(success_evaluator, "__name__", "external_evaluator")
        elif getattr(task, "ground_truth", None) is not None:
            verifier_type = "ground_truth"
        else:
            verifier_type = None

        failure_reason = result.get("failure_reason")
        if failure_reason is None:
            if not execution_success:
                failure_reason = "execution_failed"
            elif not scored:
                failure_reason = "missing_verifier"
            elif task_success is False:
                failure_reason = "verifier_rejected"

        metrics.skill_selection_f1 += selection_f1
        metrics.sequence_accuracy += int(sequence_correct)
        metrics.avg_skill_calls += calls
        metrics.skill_context_tokens += context_tokens
        metrics.prompt_tokens += prompt_tokens
        metrics.completion_tokens += completion_tokens
        metrics.total_tokens += task_tokens
        metrics.avg_execution_steps += execution_steps
        metrics.avg_execution_time_ms += execution_time_ms
        total_redundant += redundant
        total_calls += calls
        metrics.per_task.append({
            "task_id": task.id,
            "instruction": getattr(task, "instruction", ""),
            "ground_truth": getattr(task, "ground_truth", None),
            "retrieved_skill_ids": list(result.get("retrieved_skill_ids", [])),
            "retrieved_scores": list(result.get("retrieved_scores", [])),
            "retrieval_metrics": dict(result.get("retrieval_metrics", {})),
            "selected_skill_ids": selected,
            "expected_skill_ids": expected,
            "expected_skill_sequence": expected_sequence,
            "skill_selection_f1": selection_f1,
            "sequence_correct": sequence_correct,
            "execution_success": execution_success,
            "task_success": task_success,
            "scored": scored,
            "verifier_type": verifier_type,
            "failure_reason": failure_reason,
            "skill_calls": calls,
            "redundant_calls": redundant,
            "skill_context_tokens": context_tokens,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": task_tokens,
            },
            "execution_steps": execution_steps,
            "execution_time_ms": execution_time_ms,
            "answer": result.get("answer", ""),
        })

    task_count = len(tasks)
    metrics.skill_selection_f1 /= task_count
    metrics.sequence_accuracy /= task_count
    metrics.task_success_rate = (
        successful_tasks / metrics.scored_task_count
        if metrics.scored_task_count
        else None
    )
    metrics.avg_skill_calls /= task_count
    metrics.redundant_call_rate = total_redundant / total_calls if total_calls else 0.0
    metrics.skill_context_tokens /= task_count
    metrics.avg_execution_steps /= task_count
    metrics.avg_execution_time_ms /= task_count
    return metrics
