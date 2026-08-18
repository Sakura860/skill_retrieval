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
    task_success_rate: float = 0.0
    avg_skill_calls: float = 0.0
    redundant_call_rate: float = 0.0
    skill_context_tokens: float = 0.0
    total_tokens: int = 0
    avg_execution_steps: float = 0.0
    per_task: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "skill_selection_f1": self.skill_selection_f1,
            "sequence_accuracy": self.sequence_accuracy,
            "task_success_rate": self.task_success_rate,
            "avg_skill_calls": self.avg_skill_calls,
            "redundant_call_rate": self.redundant_call_rate,
            "skill_context_tokens": self.skill_context_tokens,
            "total_tokens": self.total_tokens,
            "avg_execution_steps": self.avg_execution_steps,
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


def _default_success(task: Any, result: dict) -> bool:
    if not result.get("success", False):
        return False
    ground_truth = getattr(task, "ground_truth", None)
    if ground_truth is None:
        return True
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
    success_evaluator: Callable[[Any, dict], bool] | None = None,
) -> TaskMetrics:
    """执行任务并汇总八项指标。"""
    metrics = TaskMetrics()
    if not tasks:
        return metrics

    judge = success_evaluator or _default_success
    total_redundant = 0
    total_calls = 0

    for task in tasks:
        result = run_fn(task)
        selected = list(result.get("selected_skill_ids", []))
        expected = list(getattr(task, "expected_skills", []))
        expected_sequence = list(task.gold_sequence())
        calls = int(result.get("skill_calls", len(selected)))
        redundant = _redundant_calls(selected, expected_sequence)
        success = bool(judge(task, result))
        selection_f1 = _selection_f1(selected, expected)
        sequence_correct = selected == expected_sequence
        context_tokens = int(result.get("skill_context_tokens", 0))
        task_tokens = int(result.get("token_usage", {}).get("total_tokens", 0))
        execution_steps = int(result.get("execution_steps", 0))

        metrics.skill_selection_f1 += selection_f1
        metrics.sequence_accuracy += int(sequence_correct)
        metrics.task_success_rate += int(success)
        metrics.avg_skill_calls += calls
        metrics.skill_context_tokens += context_tokens
        metrics.total_tokens += task_tokens
        metrics.avg_execution_steps += execution_steps
        total_redundant += redundant
        total_calls += calls
        metrics.per_task.append({
            "task_id": task.id,
            "selected_skill_ids": selected,
            "expected_skill_ids": expected,
            "skill_selection_f1": selection_f1,
            "sequence_correct": sequence_correct,
            "task_success": success,
            "skill_calls": calls,
            "redundant_calls": redundant,
            "skill_context_tokens": context_tokens,
            "total_tokens": task_tokens,
            "execution_steps": execution_steps,
            "answer": result.get("answer", ""),
        })

    task_count = len(tasks)
    metrics.skill_selection_f1 /= task_count
    metrics.sequence_accuracy /= task_count
    metrics.task_success_rate /= task_count
    metrics.avg_skill_calls /= task_count
    metrics.redundant_call_rate = total_redundant / total_calls if total_calls else 0.0
    metrics.skill_context_tokens /= task_count
    metrics.avg_execution_steps /= task_count
    return metrics
