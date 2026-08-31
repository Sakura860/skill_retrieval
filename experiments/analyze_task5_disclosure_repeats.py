"""Validate and aggregate repeated preregistered disclosure runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="always_full")
    parser.add_argument("--candidate", default="adaptive_signals")
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.inputs]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for field in ("protocol_sha256", "source_manifest", "task_ids", "split"):
        if any(run[field] != runs[0][field] for run in runs[1:]):
            raise ValueError(f"重复实验不一致: {field}")
    methods = (args.baseline, args.candidate)
    for run in runs:
        for method in methods:
            data = run["runs"][method]
            if data["api_errors"] or len(data["per_task"]) != len(run["task_ids"]):
                raise ValueError(f"{run['experiment_id']} {method} 不完整")

    result = {
        "protocol_sha256": runs[0]["protocol_sha256"],
        "source_manifest": runs[0]["source_manifest"],
        "task_ids": runs[0]["task_ids"],
        "repeat_count": len(runs),
        "inputs": [str(path) for path in paths],
        "methods": {},
        "paired": {},
    }
    for method in methods:
        summaries = [run["runs"][method]["summary"] for run in runs]
        result["methods"][method] = {
            "task_success_rate": mean_sd([
                item["agent"]["task_success_rate"] for item in summaries
            ]),
            "total_tokens": mean_sd([
                item["efficiency"]["total_tokens"] for item in summaries
            ]),
            "p50_end_to_end_ms": mean_sd([
                item["efficiency"]["end_to_end_time_p50_ms"]
                for item in summaries
            ]),
            "resolved_level_counts": [
                item["disclosure"]["resolved_level_counts"]
                for item in summaries
            ],
        }
    token_reductions = []
    latency_reductions = []
    decision_statuses = []
    for run in runs:
        full = run["runs"][args.baseline]["summary"]["efficiency"]
        adaptive = run["runs"][args.candidate]["summary"]["efficiency"]
        token_reductions.append(
            1 - adaptive["total_tokens"] / full["total_tokens"]
        )
        latency_reductions.append(
            1
            - adaptive["end_to_end_time_p50_ms"]
            / full["end_to_end_time_p50_ms"]
        )
        decision_key = (
            "two_stage_vs_one_stage"
            if args.baseline == "one_stage"
            else "adaptive_vs_always_full"
        )
        decision_statuses.append(run["decision_report"][decision_key]["status"])
    result["paired"] = {
        "token_reduction": mean_sd(token_reductions),
        "p50_latency_reduction": mean_sd(latency_reductions),
        "per_repeat_token_reduction": token_reductions,
        "per_repeat_p50_latency_reduction": latency_reductions,
        "decision_statuses": decision_statuses,
        "all_repeats_pass": all(status == "pass" for status in decision_statuses),
        "baseline": args.baseline,
        "candidate": args.candidate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
