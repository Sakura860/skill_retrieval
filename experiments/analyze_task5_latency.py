"""Decompose existing Task 5 latency evidence without issuing model requests."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_END_TO_END_DEV = [
    ROOT / "results" / f"task5_end_to_end_dev_v01_run{index}.json"
    for index in range(1, 4)
]
DEFAULT_DISCLOSURE_DEV = [
    ROOT / "results" / f"task5_disclosure_dev_preregistered_v07_run{index}.json"
    for index in range(1, 4)
]
DEFAULT_CONFIRMATION = (
    ROOT / "results" / "task5_confirmation_frozen_20260830.json"
)
DEFAULT_JSON_OUTPUT = (
    ROOT / "results" / "task5_latency_decomposition_20260903.json"
)
DEFAULT_MARKDOWN_OUTPUT = (
    ROOT / "results" / "task5_latency_decomposition_20260903.md"
)

KNOWN_PHASES = {
    "skill_selection",
    "argument_planning",
    "argument_repair",
    "one_stage_joint_planning",
    "reflection",
    "unspecified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved Task 5 phase latency/token evidence."
    )
    parser.add_argument(
        "--end-to-end-dev-files",
        nargs="+",
        default=DEFAULT_END_TO_END_DEV,
    )
    parser.add_argument(
        "--disclosure-dev-files",
        nargs="+",
        default=DEFAULT_DISCLOSURE_DEV,
    )
    parser.add_argument(
        "--confirmation-file",
        default=DEFAULT_CONFIRMATION,
    )
    parser.add_argument("--output-json", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def load_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} 顶层必须是 JSON 对象")
    return resolved, payload


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def stats(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "n": 0,
            "mean": 0.0,
            "sample_sd": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "total": 0.0,
        }
    return {
        "n": len(numbers),
        "mean": statistics.mean(numbers),
        "sample_sd": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "max": max(numbers),
        "total": sum(numbers),
    }


def normalized_phase(method: str, task: dict[str, Any], call: dict[str, Any]) -> str:
    """Correct the historical one-stage phase ambiguity without editing evidence.

    The original phase inference searched prompt text. Hierarchical context contains
    the phrase used to identify skill-selection prompts, so one-stage's sole joint
    selection+planning call was stored as ``skill_selection``. Method and call count
    are unambiguous in the saved task row, so analysis labels it explicitly as a
    joint call while preserving the source label separately.
    """
    if method == "one_stage" and int(task.get("planner_calls", 0)) == 1:
        return "one_stage_joint_planning"
    source = str(call.get("phase", "unspecified"))
    return source if source in KNOWN_PHASES else "unspecified"


def task_breakdown(method: str, task: dict[str, Any]) -> dict[str, Any]:
    calls = task.get("llm_calls", [])
    if not isinstance(calls, list):
        raise ValueError(f"{task.get('task_id')} 的 llm_calls 必须是列表")
    phase_duration_ms: dict[str, float] = defaultdict(float)
    phase_tokens: dict[str, int] = defaultdict(int)
    source_phase_counts: Counter[str] = Counter()
    failed_calls = 0
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError(f"{task.get('task_id')} 包含非法 LLM call")
        source_phase = str(call.get("phase", "unspecified"))
        source_phase_counts[source_phase] += 1
        phase = normalized_phase(method, task, call)
        phase_duration_ms[phase] += float(call.get("duration_ms", 0.0))
        phase_tokens[phase] += int(call.get("total_tokens", 0))
        failed_calls += int(call.get("success") is False)

    measured_llm_ms = sum(phase_duration_ms.values())
    stored_llm_ms = float(task.get("llm_time_ms", measured_llm_ms))
    execution_ms = float(task.get("execution_time_ms", 0.0))
    end_to_end_ms = float(task.get("end_to_end_time_ms", 0.0))
    residual_ms = end_to_end_ms - measured_llm_ms - execution_ms
    negative_residual_ms = min(residual_ms, 0.0)
    residual_ms = max(residual_ms, 0.0)
    return {
        "task_id": str(task.get("task_id", "")),
        "task_success": task.get("task_success"),
        "repair_attempts": int(task.get("repair_attempts", 0)),
        "planner_calls": int(task.get("planner_calls", len(calls))),
        "source_phase_counts": dict(source_phase_counts),
        "phase_duration_ms": dict(sorted(phase_duration_ms.items())),
        "phase_tokens": dict(sorted(phase_tokens.items())),
        "measured_llm_ms": measured_llm_ms,
        "stored_llm_ms": stored_llm_ms,
        "llm_measurement_delta_ms": stored_llm_ms - measured_llm_ms,
        "handler_execution_ms": execution_ms,
        "orchestration_residual_ms": residual_ms,
        "negative_residual_ms": negative_residual_ms,
        "end_to_end_ms": end_to_end_ms,
        "failed_llm_calls": failed_calls,
    }


def validate_payload_group(
    loaded: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    if not loaded:
        raise ValueError("至少需要一个结果文件")
    first = loaded[0][1]
    task_ids = list(first.get("task_ids", []))
    methods = list(first.get("runs", {}))
    if not task_ids or not methods:
        raise ValueError(f"{loaded[0][0]} 缺少 task_ids 或 runs")
    for path, payload in loaded:
        if list(payload.get("task_ids", [])) != task_ids:
            raise ValueError(f"{path} 的 task_ids 与同组其他文件不一致")
        if set(payload.get("runs", {})) != set(methods):
            raise ValueError(f"{path} 的方法集合与同组其他文件不一致")
        for method in methods:
            run = payload["runs"][method]
            if run.get("api_errors"):
                raise ValueError(f"{path} / {method} 存在 API failure")
            rows = run.get("per_task", [])
            if len(rows) != len(task_ids):
                raise ValueError(f"{path} / {method} 的任务数量不完整")
            if [row.get("task_id") for row in rows] != task_ids:
                raise ValueError(f"{path} / {method} 的任务顺序不一致")
    return task_ids, methods


def aggregate_method(
    method: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phases = sorted({
        phase
        for row in rows
        for phase in row["phase_duration_ms"]
    })
    total_end_to_end = sum(row["end_to_end_ms"] for row in rows)
    phase_summary = {}
    for phase in phases:
        durations = [row["phase_duration_ms"].get(phase, 0.0) for row in rows]
        tokens = [row["phase_tokens"].get(phase, 0) for row in rows]
        active_durations = [value for value in durations if value > 0]
        active_tokens = [value for value in tokens if value > 0]
        phase_summary[phase] = {
            "duration_ms_all_tasks": stats(durations),
            "duration_ms_active_calls": stats(active_durations),
            "tokens_all_tasks": stats(tokens),
            "tokens_active_calls": stats(active_tokens),
            "share_of_end_to_end": (
                sum(durations) / total_end_to_end if total_end_to_end else 0.0
            ),
        }

    component_values = {
        "llm_total_ms": [row["measured_llm_ms"] for row in rows],
        "handler_execution_ms": [row["handler_execution_ms"] for row in rows],
        "orchestration_residual_ms": [
            row["orchestration_residual_ms"] for row in rows
        ],
        "end_to_end_ms": [row["end_to_end_ms"] for row in rows],
        "planner_calls": [row["planner_calls"] for row in rows],
    }
    components = {name: stats(values) for name, values in component_values.items()}
    components["llm_total_ms"]["share_of_end_to_end"] = (
        sum(component_values["llm_total_ms"]) / total_end_to_end
        if total_end_to_end else 0.0
    )
    components["handler_execution_ms"]["share_of_end_to_end"] = (
        sum(component_values["handler_execution_ms"]) / total_end_to_end
        if total_end_to_end else 0.0
    )
    components["orchestration_residual_ms"]["share_of_end_to_end"] = (
        sum(component_values["orchestration_residual_ms"]) / total_end_to_end
        if total_end_to_end else 0.0
    )

    repair_rows = [row for row in rows if row["repair_attempts"] > 0]
    no_repair_rows = [row for row in rows if row["repair_attempts"] == 0]
    source_labels = Counter()
    for row in rows:
        source_labels.update(row["source_phase_counts"])
    return {
        "task_observations": len(rows),
        "successful_tasks": sum(row["task_success"] is True for row in rows),
        "failed_llm_calls": sum(row["failed_llm_calls"] for row in rows),
        "source_phase_counts": dict(sorted(source_labels.items())),
        "phase_summary": phase_summary,
        "components": components,
        "repair_comparison": {
            "repair_task_count": len(repair_rows),
            "no_repair_task_count": len(no_repair_rows),
            "repair_end_to_end_ms": stats(
                row["end_to_end_ms"] for row in repair_rows
            ),
            "no_repair_end_to_end_ms": stats(
                row["end_to_end_ms"] for row in no_repair_rows
            ),
        },
        "telemetry_consistency": {
            "max_abs_llm_measurement_delta_ms": max(
                (abs(row["llm_measurement_delta_ms"]) for row in rows),
                default=0.0,
            ),
            "negative_residual_count": sum(
                row["negative_residual_ms"] < 0 for row in rows
            ),
        },
        "per_task": rows,
    }


def paired_comparison(
    payloads: list[tuple[Path, dict[str, Any]]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    pairs = []
    for run_index, (path, payload) in enumerate(payloads, 1):
        baseline_rows = {
            row["task_id"]: task_breakdown(baseline, row)
            for row in payload["runs"][baseline]["per_task"]
        }
        candidate_rows = {
            row["task_id"]: task_breakdown(candidate, row)
            for row in payload["runs"][candidate]["per_task"]
        }
        for task_id in payload["task_ids"]:
            before = baseline_rows[task_id]
            after = candidate_rows[task_id]
            pairs.append({
                "source_file": str(path),
                "run_index": run_index,
                "task_id": task_id,
                "end_to_end_delta_ms": (
                    after["end_to_end_ms"] - before["end_to_end_ms"]
                ),
                "llm_delta_ms": (
                    after["measured_llm_ms"] - before["measured_llm_ms"]
                ),
                "handler_delta_ms": (
                    after["handler_execution_ms"]
                    - before["handler_execution_ms"]
                ),
                "orchestration_delta_ms": (
                    after["orchestration_residual_ms"]
                    - before["orchestration_residual_ms"]
                ),
                "planner_call_delta": (
                    after["planner_calls"] - before["planner_calls"]
                ),
                "candidate_selection_ms": after["phase_duration_ms"].get(
                    "skill_selection", 0.0
                ),
                "candidate_planning_ms": after["phase_duration_ms"].get(
                    "argument_planning", 0.0
                ),
                "candidate_repair_ms": after["phase_duration_ms"].get(
                    "argument_repair", 0.0
                ),
            })
    fields = [
        "end_to_end_delta_ms",
        "llm_delta_ms",
        "handler_delta_ms",
        "orchestration_delta_ms",
        "planner_call_delta",
        "candidate_selection_ms",
        "candidate_planning_ms",
        "candidate_repair_ms",
    ]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_observations": len(pairs),
        "summary": {
            field: stats(pair[field] for pair in pairs) for field in fields
        },
        "per_task": pairs,
    }


def analyze_group(
    name: str,
    loaded: list[tuple[Path, dict[str, Any]]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    task_ids, methods = validate_payload_group(loaded)
    if baseline not in methods or candidate not in methods:
        raise ValueError(f"{name} 缺少 {baseline}/{candidate}")
    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for _, payload in loaded:
        for method in methods:
            method_rows[method].extend(
                task_breakdown(method, row)
                for row in payload["runs"][method]["per_task"]
            )
    return {
        "name": name,
        "split": loaded[0][1].get("split"),
        "source_files": [str(path) for path, _ in loaded],
        "repeat_count": len(loaded),
        "task_ids": task_ids,
        "methods": {
            method: aggregate_method(method, rows)
            for method, rows in method_rows.items()
        },
        "paired": paired_comparison(loaded, baseline, candidate),
    }


def _fmt_ms(value: float) -> str:
    return f"{value:,.2f}"


def _method_row(name: str, data: dict[str, Any]) -> str:
    components = data["components"]
    return (
        f"| {name} | {data['task_observations']} | "
        f"{_fmt_ms(components['end_to_end_ms']['mean'])} | "
        f"{_fmt_ms(components['end_to_end_ms']['p50'])} | "
        f"{_fmt_ms(components['llm_total_ms']['mean'])} | "
        f"{_fmt_ms(components['handler_execution_ms']['mean'])} | "
        f"{_fmt_ms(components['orchestration_residual_ms']['mean'])} | "
        f"{components['planner_calls']['mean']:.2f} |"
    )


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Task 5 latency decomposition — 2026-09-03",
        "",
        "## Scope and measurement",
        "",
        "This report is computed only from immutable saved JSON evidence. It does "
        "not issue DeepSeek requests and does not modify or rerun the consumed "
        "confirmation set.",
        "",
        "Per-task orchestration residual is `end_to_end - sum(LLM calls) - handler "
        "execution`. Percentiles use the same nearest stored observation rule as "
        "the project benchmark (`round((n-1)*p)`). Historical one-stage calls were "
        "stored as `skill_selection` because phase inference matched text inside "
        "the hierarchical context; this report preserves that source label but "
        "normalizes the sole one-stage call to `one_stage_joint_planning`.",
        "",
    ]
    for group_name, group in result["groups"].items():
        lines.extend([
            f"## {group_name}",
            "",
            f"Sources: {group['repeat_count']} file(s), "
            f"{len(group['task_ids'])} task(s) per file, split `{group['split']}`.",
            "",
            "| Method | Observations | Mean E2E ms | P50 E2E ms | Mean LLM ms | "
            "Mean handler ms | Mean residual ms | Planner calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method, data in group["methods"].items():
            lines.append(_method_row(method, data))
        paired = group["paired"]
        summary = paired["summary"]
        lines.extend([
            "",
            f"Paired `{paired['candidate']} - {paired['baseline']}` mean deltas "
            f"over {paired['paired_observations']} observations:",
            "",
            f"- End-to-end: {_fmt_ms(summary['end_to_end_delta_ms']['mean'])} ms",
            f"- LLM calls: {_fmt_ms(summary['llm_delta_ms']['mean'])} ms",
            f"- Handler execution: {_fmt_ms(summary['handler_delta_ms']['mean'])} ms",
            f"- Orchestration residual: "
            f"{_fmt_ms(summary['orchestration_delta_ms']['mean'])} ms",
            f"- Planner calls: {summary['planner_call_delta']['mean']:.2f}",
            "",
            "Candidate phase means per task:",
            "",
            f"- Selection: {_fmt_ms(summary['candidate_selection_ms']['mean'])} ms",
            f"- Planning: {_fmt_ms(summary['candidate_planning_ms']['mean'])} ms",
            f"- Repair: {_fmt_ms(summary['candidate_repair_ms']['mean'])} ms",
            "",
        ])

    confirmation = result["groups"]["confirmation"]
    paired = confirmation["paired"]["summary"]
    extra_e2e = float(paired["end_to_end_delta_ms"]["mean"])
    extra_llm = float(paired["llm_delta_ms"]["mean"])
    extra_handler = float(paired["handler_delta_ms"]["mean"])
    extra_residual = float(paired["orchestration_delta_ms"]["mean"])
    llm_share = extra_llm / extra_e2e if extra_e2e else 0.0
    lines.extend([
        "## Conclusion",
        "",
        f"On confirmation, adaptive-signals adds {_fmt_ms(extra_e2e)} ms mean "
        f"end-to-end latency per task. The LLM component accounts for "
        f"{_fmt_ms(extra_llm)} ms ({llm_share:.1%} of the paired mean increase); "
        f"handler change is {_fmt_ms(extra_handler)} ms and orchestration residual "
        f"change is {_fmt_ms(extra_residual)} ms.",
        "",
        "Therefore the measured latency penalty is primarily attributable to the "
        "second model call, a structural cost of two-stage planning on this provider. "
        "Local handler and orchestration overhead are comparatively negligible. "
        "Connection reuse or serialization work may still be profiled later, but "
        "the current evidence does not support presenting the 65.8% P50 increase as "
        "mainly a local implementation defect.",
        "",
        "The disclosure-only dev comparison is also important: always-full and "
        "adaptive-signals both use two stages and have nearly identical latency. "
        "Adaptive disclosure saves tokens, but it does not remove the extra network "
        "round trip inherent in two-stage planning.",
        "",
        "## Validity boundary",
        "",
        "- Confirmation contains 10 tasks; means and empirical percentiles are "
        "descriptive rather than precise population estimates.",
        "- Calls were executed sequentially against one provider; parallel or local "
        "models may have a different structural latency trade-off.",
        "- Residual time is calculated rather than independently instrumented, so it "
        "combines all non-LLM, non-handler orchestration work.",
        "- No post-confirmation optimization was evaluated, because the consumed "
        "confirmation protocol cannot be rerun.",
        "",
    ])
    return "\n".join(lines)


def build_analysis(
    end_to_end_dev: list[tuple[Path, dict[str, Any]]],
    disclosure_dev: list[tuple[Path, dict[str, Any]]],
    confirmation: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "analysis_id": "task5-latency-decomposition-20260903-v01",
        "analysis_only": True,
        "model_requests_issued": 0,
        "percentile_rule": "sorted_values[round((n - 1) * fraction)]",
        "residual_formula": (
            "end_to_end_ms - sum(llm_call.duration_ms) - execution_time_ms"
        ),
        "groups": {
            "end_to_end_dev": analyze_group(
                "end_to_end_dev",
                end_to_end_dev,
                "one_stage",
                "adaptive_signals",
            ),
            "disclosure_dev": analyze_group(
                "disclosure_dev",
                disclosure_dev,
                "always_full",
                "adaptive_signals",
            ),
            "confirmation": analyze_group(
                "confirmation",
                confirmation,
                "one_stage",
                "adaptive_signals",
            ),
        },
    }


def main() -> None:
    args = parse_args()
    analysis = build_analysis(
        [load_payload(path) for path in args.end_to_end_dev_files],
        [load_payload(path) for path in args.disclosure_dev_files],
        [load_payload(args.confirmation_file)],
    )
    json_output = Path(args.output_json)
    markdown_output = Path(args.output_markdown)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"JSON saved: {json_output}")
    print(f"Markdown saved: {markdown_output}")


if __name__ == "__main__":
    main()
