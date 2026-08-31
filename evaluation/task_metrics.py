"""Agent 任务级评测。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evaluation.verifiers import TaskVerifierRegistry, VerifierResult, verify_task


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
    avg_llm_time_ms: float = 0.0
    avg_end_to_end_time_ms: float = 0.0
    avg_planner_calls: float = 0.0
    avg_argument_repairs: float = 0.0
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
            "avg_llm_time_ms": self.avg_llm_time_ms,
            "avg_end_to_end_time_ms": self.avg_end_to_end_time_ms,
            "avg_planner_calls": self.avg_planner_calls,
            "avg_argument_repairs": self.avg_argument_repairs,
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
        llm_times = sorted(float(item.get("llm_time_ms", 0.0)) for item in self.per_task)
        end_to_end_times = sorted(
            float(item.get("end_to_end_time_ms", 0.0)) for item in self.per_task
        )

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            return values[round((len(values) - 1) * fraction)]

        phase_time: dict[str, float] = {}
        phase_tokens: dict[str, int] = {}
        for task in self.per_task:
            for call in task.get("llm_calls", []):
                phase = str(call.get("phase", "unspecified"))
                phase_time[phase] = phase_time.get(phase, 0.0) + float(
                    call.get("duration_ms", 0.0)
                )
                phase_tokens[phase] = phase_tokens.get(phase, 0) + int(
                    call.get("total_tokens", 0)
                )
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
            "avg_llm_time_ms": self.avg_llm_time_ms,
            "llm_time_p50_ms": percentile(llm_times, 0.5),
            "llm_time_p95_ms": percentile(llm_times, 0.95),
            "avg_end_to_end_time_ms": self.avg_end_to_end_time_ms,
            "end_to_end_time_p50_ms": percentile(end_to_end_times, 0.5),
            "end_to_end_time_p95_ms": percentile(end_to_end_times, 0.95),
            "llm_phase_time_ms": phase_time,
            "llm_phase_tokens": phase_tokens,
            "avg_planner_calls": self.avg_planner_calls,
            "avg_argument_repairs": self.avg_argument_repairs,
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


def evaluate_agent(
    run_fn: Callable[[Any], dict],
    tasks: list[Any],
    success_evaluator: Callable[[Any, dict], bool | None] | None = None,
    verifier_registry: TaskVerifierRegistry | None = None,
) -> TaskMetrics:
    """执行任务并汇总 Agent、任务成功与效率指标。"""
    metrics = TaskMetrics(task_count=len(tasks))
    if not tasks:
        return metrics

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
        if success_evaluator is not None:
            judged = success_evaluator(task, result)
            verification = None if judged is None else VerifierResult(
                passed=bool(judged),
                verifier_type=getattr(
                    success_evaluator,
                    "__name__",
                    "external_evaluator",
                ),
                failure_reason=None if judged else "verifier_rejected",
            )
        else:
            verification = verify_task(task, result, verifier_registry)
        scored = verification is not None
        execution_success = bool(result.get("success", False))
        task_success = (
            execution_success and verification.passed
            if verification is not None
            else None
        )
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
        llm_time_ms = float(result.get("llm_time_ms", 0.0))
        end_to_end_time_ms = float(result.get("end_to_end_time_ms", 0.0))
        planner_calls = int(result.get("planner_calls", 0))
        repair_attempts = int(result.get("repair_attempts", 0))
        verifier_type = verification.verifier_type if verification else None

        failure_reason = result.get("failure_reason")
        if failure_reason is None:
            if not execution_success:
                failure_reason = "execution_failed"
            elif not scored:
                failure_reason = "missing_verifier"
            elif task_success is False:
                failure_reason = verification.failure_reason or "verifier_rejected"

        metrics.skill_selection_f1 += selection_f1
        metrics.sequence_accuracy += int(sequence_correct)
        metrics.avg_skill_calls += calls
        metrics.skill_context_tokens += context_tokens
        metrics.prompt_tokens += prompt_tokens
        metrics.completion_tokens += completion_tokens
        metrics.total_tokens += task_tokens
        metrics.avg_execution_steps += execution_steps
        metrics.avg_execution_time_ms += execution_time_ms
        metrics.avg_llm_time_ms += llm_time_ms
        metrics.avg_end_to_end_time_ms += end_to_end_time_ms
        metrics.avg_planner_calls += planner_calls
        metrics.avg_argument_repairs += repair_attempts
        total_redundant += redundant
        total_calls += calls
        metrics.per_task.append({
            "task_id": task.id,
            "instruction": getattr(task, "instruction", ""),
            "ground_truth": getattr(task, "ground_truth", None),
            "retrieved_skill_ids": list(result.get("retrieved_skill_ids", [])),
            "retrieved_scores": list(result.get("retrieved_scores", [])),
            "raw_bm25_skill_ids": list(result.get("raw_bm25_skill_ids", [])),
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
            "verifier_details": verification.details if verification else {},
            "verifier_expected": verification.expected if verification else None,
            "verifier_actual": verification.actual if verification else None,
            "failure_reason": failure_reason,
            "skill_calls": calls,
            "redundant_calls": redundant,
            "skill_context_tokens": context_tokens,
            "context_budget_tokens": result.get("context_budget_tokens"),
            "planner_mode": result.get("planner_mode", "one_stage"),
            "planner_disclosure_level": result.get(
                "planner_disclosure_level", "full"
            ),
            "resolved_disclosure_level": result.get(
                "resolved_disclosure_level",
                result.get("planner_disclosure_level", "full"),
            ),
            "requested_disclosure_levels": list(
                result.get("requested_disclosure_levels", [])
            ),
            "disclosure_reasons": list(result.get("disclosure_reasons", [])),
            "escalation_reasons": list(result.get("escalation_reasons", [])),
            "disclosure_signals": dict(result.get("disclosure_signals", {})),
            "planner_calls": planner_calls,
            "repair_attempts": repair_attempts,
            "validation_errors": list(result.get("validation_errors", [])),
            "selection_evidence": list(result.get("selection_evidence", [])),
            "selection_context_tokens": int(
                result.get("selection_context_tokens", 0)
            ),
            "planning_context_tokens": int(
                result.get("planning_context_tokens", 0)
            ),
            "candidate_count": result.get("candidate_count"),
            "gold_augmented_skill_ids": list(
                result.get("gold_augmented_skill_ids", [])
            ),
            "target_gold_rank": result.get("target_gold_rank"),
            "exposed_skill_ids": list(result.get("exposed_skill_ids", [])),
            "detailed_skill_ids": list(result.get("detailed_skill_ids", [])),
            "truncated_skill_ids": list(result.get("truncated_skill_ids", [])),
            "added_skill_ids": list(result.get("added_skill_ids", [])),
            "graph_issues": list(result.get("graph_issues", [])),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": task_tokens,
            },
            "execution_steps": execution_steps,
            "execution_time_ms": execution_time_ms,
            "llm_time_ms": llm_time_ms,
            "end_to_end_time_ms": end_to_end_time_ms,
            "llm_calls": list(result.get("llm_calls", [])),
            "answer": result.get("answer", ""),
            "trajectory": list(result.get("trajectory", [])),
            "initial_state": result.get("initial_state"),
            "final_state": result.get("final_state"),
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
    metrics.avg_llm_time_ms /= task_count
    metrics.avg_end_to_end_time_ms /= task_count
    metrics.avg_planner_calls /= task_count
    metrics.avg_argument_repairs /= task_count
    return metrics
