"""Offline regression tests for Task 5 latency decomposition."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_task5_latency import (  # noqa: E402
    aggregate_method,
    analyze_group,
    normalized_phase,
    percentile,
    task_breakdown,
)


def task_row(
    task_id: str,
    *,
    planner_mode: str,
    phases: list[tuple[str, float, int]],
    execution_ms: float = 2.0,
    residual_ms: float = 3.0,
    repairs: int = 0,
) -> dict:
    calls = [
        {
            "phase": phase,
            "duration_ms": duration,
            "total_tokens": tokens,
            "success": True,
        }
        for phase, duration, tokens in phases
    ]
    llm_ms = sum(call["duration_ms"] for call in calls)
    return {
        "task_id": task_id,
        "task_success": True,
        "planner_mode": planner_mode,
        "planner_calls": len(calls),
        "repair_attempts": repairs,
        "llm_calls": calls,
        "llm_time_ms": llm_ms,
        "execution_time_ms": execution_ms,
        "end_to_end_time_ms": llm_ms + execution_ms + residual_ms,
    }


def payload() -> dict:
    one_stage = task_row(
        "t1",
        planner_mode="one_stage",
        phases=[("skill_selection", 100.0, 50)],
    )
    adaptive = task_row(
        "t1",
        planner_mode="two_stage",
        phases=[
            ("skill_selection", 80.0, 30),
            ("argument_planning", 90.0, 35),
        ],
    )
    return {
        "split": "dev",
        "task_ids": ["t1"],
        "runs": {
            "one_stage": {
                "per_task": [one_stage],
                "api_errors": [],
            },
            "adaptive_signals": {
                "per_task": [adaptive],
                "api_errors": [],
            },
        },
    }


def main() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 3

    raw_one_stage = payload()["runs"]["one_stage"]["per_task"][0]
    assert normalized_phase(
        "one_stage", raw_one_stage, raw_one_stage["llm_calls"][0]
    ) == "one_stage_joint_planning"

    breakdown = task_breakdown("one_stage", raw_one_stage)
    assert breakdown["phase_duration_ms"] == {
        "one_stage_joint_planning": 100.0
    }
    assert breakdown["orchestration_residual_ms"] == 3.0
    assert breakdown["source_phase_counts"] == {"skill_selection": 1}

    aggregate = aggregate_method("one_stage", [breakdown])
    assert aggregate["components"]["llm_total_ms"]["mean"] == 100.0
    assert aggregate["components"]["handler_execution_ms"]["mean"] == 2.0
    assert aggregate["components"]["orchestration_residual_ms"]["mean"] == 3.0

    group = analyze_group(
        "synthetic",
        [(Path("run1.json"), payload())],
        "one_stage",
        "adaptive_signals",
    )
    paired = group["paired"]["summary"]
    assert paired["end_to_end_delta_ms"]["mean"] == 70.0
    assert paired["llm_delta_ms"]["mean"] == 70.0
    assert paired["handler_delta_ms"]["mean"] == 0.0
    assert paired["orchestration_delta_ms"]["mean"] == 0.0
    assert paired["candidate_selection_ms"]["mean"] == 80.0
    assert paired["candidate_planning_ms"]["mean"] == 90.0

    broken = payload()
    broken["runs"]["adaptive_signals"]["per_task"] = []
    try:
        analyze_group(
            "broken",
            [(Path("broken.json"), broken)],
            "one_stage",
            "adaptive_signals",
        )
    except ValueError as exc:
        assert "任务数量不完整" in str(exc)
    else:
        raise AssertionError("incomplete latency evidence should be rejected")

    print("task5 latency analysis tests passed")


if __name__ == "__main__":
    main()
