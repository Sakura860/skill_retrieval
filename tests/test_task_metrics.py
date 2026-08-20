"""八项任务指标测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.schemas import Task
from evaluation.task_metrics import evaluate_agent


def test_metrics():
    tasks = [
        Task("t1", "task 1", ["s1", "s2"], ["s1", "s2"], "ok"),
        Task("t2", "task 2", ["s3"], ["s3"], "expected"),
    ]
    results = {
        "t1": {
            "success": True,
            "answer": "ok",
            "selected_skill_ids": ["s1", "s2"],
            "skill_calls": 2,
            "skill_context_tokens": 100,
            "token_usage": {"total_tokens": 50},
            "execution_steps": 2,
        },
        "t2": {
            "success": False,
            "answer": "failed",
            "selected_skill_ids": ["s4", "s4"],
            "skill_calls": 2,
            "skill_context_tokens": 60,
            "token_usage": {"total_tokens": 20},
            "execution_steps": 2,
        },
    }

    metrics = evaluate_agent(lambda task: results[task.id], tasks)
    assert metrics.skill_selection_f1 == 0.5
    assert metrics.sequence_accuracy == 0.5
    assert metrics.task_success_rate == 0.5
    assert metrics.avg_skill_calls == 2.0
    assert metrics.redundant_call_rate == 0.5
    assert metrics.skill_context_tokens == 80.0
    assert metrics.total_tokens == 70
    assert metrics.avg_execution_steps == 2.0
    assert metrics.scored_task_count == 2
    assert metrics.unscored_task_count == 0


def test_unscored_task_is_not_counted_as_success():
    task = Task("t1", "task without verifier", ["s1"], ["s1"])
    result = {
        "success": True,
        "answer": "looks fine",
        "selected_skill_ids": ["s1"],
    }

    metrics = evaluate_agent(lambda _: result, [task])

    assert metrics.task_success_rate is None
    assert metrics.scored_task_count == 0
    assert metrics.unscored_task_count == 1
    assert metrics.per_task[0]["execution_success"] is True
    assert metrics.per_task[0]["task_success"] is None
    assert metrics.per_task[0]["failure_reason"] == "missing_verifier"


def test_execution_success_does_not_override_verifier_failure():
    task = Task("t1", "calculate", ["s1"], ["s1"], 231)
    result = {
        "success": True,
        "answer": "230",
        "selected_skill_ids": ["s1"],
    }

    metrics = evaluate_agent(lambda _: result, [task])

    assert metrics.task_success_rate == 0.0
    assert metrics.per_task[0]["execution_success"] is True
    assert metrics.per_task[0]["task_success"] is False
    assert metrics.per_task[0]["failure_reason"] == "output_mismatch"
    assert metrics.per_task[0]["verifier_type"] == "ground_truth"
    assert metrics.per_task[0]["verifier_expected"] == 231
    assert metrics.per_task[0]["verifier_actual"] == "230"


if __name__ == "__main__":
    test_metrics()
    test_unscored_task_is_not_counted_as_success()
    test_execution_success_does_not_override_verifier_failure()
    print("task metrics test passed")
