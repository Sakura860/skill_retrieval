"""Confirmation-set governance and decision-report regressions."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_task5_disclosure import (  # noqa: E402
    complete_confirmation_run,
    decision_report,
    reserve_confirmation_run,
)


def protocol() -> dict:
    return {
        "protocol_id": "confirmation-guard-test",
        "confirmation_policy": {"maximum_runs": 1},
        "decision_rules": {
            "two_stage_vs_one_stage": {
                "candidate_method": "adaptive_signals",
                "minimum_absolute_task_success_gain": 0.1,
                "maximum_total_token_ratio": 1.15,
                "maximum_p50_end_to_end_latency_ratio": 1.8,
            },
            "adaptive_vs_always_full": {
                "maximum_absolute_task_success_drop": 0.02,
                "minimum_total_token_reduction": 0.1,
                "minimum_p50_end_to_end_latency_reduction": 0.15,
            },
        },
    }


def summary(success: float, tokens: int, latency: float) -> dict:
    return {
        "summary": {
            "agent": {"task_success_rate": success},
            "efficiency": {
                "total_tokens": tokens,
                "end_to_end_time_p50_ms": latency,
            },
        }
    }


def test_confirmation_slot_is_single_and_resumable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = root / "registry.json"
        output = root / "confirmation.json"
        current = protocol()
        reserve_confirmation_run(
            current, "sha", output, "run-1", False, registry
        )
        try:
            reserve_confirmation_run(
                current, "sha", root / "other.json", "run-2", False, registry
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("a second confirmation run must be refused")

        reserve_confirmation_run(current, "sha", output, "ignored", True, registry)
        output.write_text("{}", encoding="utf-8")
        complete_confirmation_run(current, "sha", output, registry)
        stored = json.loads(registry.read_text(encoding="utf-8"))["runs"]
        assert len(stored) == 1
        assert stored[0]["status"] == "completed"
        assert stored[0]["result_sha256"]


def test_missing_always_full_is_not_self_comparison() -> None:
    output = {
        "protocol_snapshot": protocol(),
        "runs": {
            "one_stage": summary(0.7, 1000, 1000),
            "adaptive_signals": summary(0.8, 1100, 1500),
        },
    }
    report = decision_report(output)
    assert report["two_stage_vs_one_stage"]["status"] == "pass"
    assert report["adaptive_vs_always_full"]["status"] == "incomplete"


def test_exact_ten_point_gain_survives_float_roundoff() -> None:
    output = {
        "protocol_snapshot": protocol(),
        "runs": {
            "one_stage": summary(0.9, 8469, 1932.965),
            "adaptive_signals": summary(1.0, 8040, 3205.5484),
        },
    }
    report = decision_report(output)
    assert report["two_stage_vs_one_stage"]["observed"][
        "task_success_gain"
    ] < 0.1
    assert report["two_stage_vs_one_stage"]["status"] == "pass"


if __name__ == "__main__":
    test_confirmation_slot_is_single_and_resumable()
    test_missing_always_full_is_not_self_comparison()
    test_exact_ten_point_gain_survives_float_roundoff()
    print("task5 confirmation guard tests passed")
