"""聚合 Task 5 多次 dev 运行，报告均值、样本标准差和逐任务一致性。"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PLANNING = [
    ROOT / "results" / "task5_planning_v01_dev_dataflow_v2.json",
    ROOT / "results" / "task5_planning_v01_dev_repeat2.json",
    ROOT / "results" / "task5_planning_v01_dev_repeat3.json",
]
DEFAULT_GRAPH = [
    ROOT / "results" / "task5_graph_v02_dev.json",
    ROOT / "results" / "task5_graph_v02_dev_repeat2.json",
    ROOT / "results" / "task5_graph_v02_dev_repeat3.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-files", nargs="+", default=DEFAULT_PLANNING)
    parser.add_argument("--graph-files", nargs="+", default=DEFAULT_GRAPH)
    parser.add_argument(
        "--output",
        default=ROOT / "results" / "task5_dev_repeat_summary_20260824.json",
    )
    return parser.parse_args()


def _load(paths: list[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded.append((path, payload))
    return loaded


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _metric(summary: dict, path: tuple[str, str]) -> float:
    return float(summary[path[0]][path[1]])


METRICS = {
    "task_success_rate": ("agent", "task_success_rate"),
    "skill_selection_f1": ("agent", "skill_selection_f1"),
    "sequence_accuracy": ("agent", "sequence_accuracy"),
    "avg_skill_context_tokens": ("efficiency", "avg_skill_context_tokens"),
    "total_tokens": ("efficiency", "total_tokens"),
    "avg_planner_calls": ("efficiency", "avg_planner_calls"),
    "avg_argument_repairs": ("efficiency", "avg_argument_repairs"),
}


def _validate_group(
    loaded: list[tuple[Path, dict[str, Any]]],
    *,
    dataset: str,
    split: str,
) -> tuple[list[str], list[str]]:
    if len(loaded) < 2:
        raise ValueError("稳定性分析至少需要两次运行")
    first = loaded[0][1]
    task_ids = first.get("task_ids", [])
    preferred_order = {
        "one_stage": 0,
        "two_stage_no_repair": 1,
        "two_stage_one_repair": 2,
        "without_completion": 0,
        "with_prerequisite_completion": 1,
    }
    methods = sorted(
        first.get("runs", {}), key=lambda item: (preferred_order.get(item, 99), item)
    )
    for path, payload in loaded:
        if payload.get("dataset") != dataset or payload.get("split") != split:
            raise ValueError(f"{path} 的 dataset/split 不匹配")
        if payload.get("task_ids") != task_ids:
            raise ValueError(f"{path} 的 task_ids 不一致")
        if set(payload.get("runs", {})) != set(methods):
            raise ValueError(f"{path} 的方法集合不一致")
        for method in methods:
            run = payload["runs"][method]
            if run.get("api_errors"):
                raise ValueError(f"{path} / {method} 存在 API failure")
            if len(run.get("per_task", [])) != len(task_ids):
                raise ValueError(f"{path} / {method} 未完成全部任务")
    return task_ids, methods


def _aggregate(
    loaded: list[tuple[Path, dict[str, Any]]],
    *,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    task_ids, methods = _validate_group(loaded, dataset=dataset, split=split)
    result: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "repeat_count": len(loaded),
        "files": [str(path) for path, _ in loaded],
        "task_ids": task_ids,
        "methods": {},
    }
    for method in methods:
        runs = [payload["runs"][method] for _, payload in loaded]
        by_task = {
            task_id: [
                next(row for row in run["per_task"] if row["task_id"] == task_id)
                for run in runs
            ]
            for task_id in task_ids
        }
        per_task = {}
        for task_id, rows in by_task.items():
            successes = [row.get("task_success") is True for row in rows]
            selections = [float(row["skill_selection_f1"]) for row in rows]
            sequences = [bool(row["sequence_correct"]) for row in rows]
            per_task[task_id] = {
                "successes": successes,
                "success_count": sum(successes),
                "selection_f1": selections,
                "sequence_correct": sequences,
                "outcome_stable": len(set(successes)) == 1,
            }
        result["methods"][method] = {
            "metrics": {
                name: _stats([
                    _metric(run["summary"], metric_path) for run in runs
                ])
                for name, metric_path in METRICS.items()
            },
            "stable_task_outcomes": sum(
                row["outcome_stable"] for row in per_task.values()
            ),
            "task_count": len(task_ids),
            "per_task": per_task,
        }
        if method == "with_prerequisite_completion":
            additions = {
                task_id: [row.get("added_skill_ids", []) for row in rows]
                for task_id, rows in by_task.items()
            }
            result["methods"][method]["added_skill_ids"] = additions
            result["methods"][method]["additions_stable"] = all(
                len({tuple(items) for items in repeats}) == 1
                for repeats in additions.values()
            )
    return result


def main() -> None:
    args = parse_args()
    output = {
        "planning": _aggregate(
            _load(args.planning_files), dataset="benchmark_v01", split="dev"
        ),
        "graph": _aggregate(
            _load(args.graph_files), dataset="benchmark_v02", split="dev"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
