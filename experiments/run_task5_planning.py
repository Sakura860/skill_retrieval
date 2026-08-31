"""任务 5：一阶段与两阶段 Planner 的受控 DeepSeek 对比。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from data.loader import load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from organization.hierarchical import HierarchicalOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

METHODS = {
    "one_stage": {
        "planner_mode": "one_stage",
        "max_argument_repairs": 0,
        "detail_top_k": 3,
    },
    "two_stage_no_repair": {
        "planner_mode": "two_stage",
        "max_argument_repairs": 0,
        "detail_top_k": 0,
    },
    "two_stage_one_repair": {
        "planner_mode": "two_stage",
        "max_argument_repairs": 1,
        "detail_top_k": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--dataset",
        default="benchmark_v01",
        choices=("benchmark_v01", "benchmark_v02"),
    )
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument(
        "--task-ids",
        default="",
        help="逗号分隔；为空时按 --split 选择任务",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"逗号分隔，可选: {','.join(METHODS)}",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mean(rows: list[dict], getter) -> float | None:
    values = [getter(row) for row in rows]
    return sum(values) / len(values) if values else None


def _summary(rows: list[dict], errors: list[dict]) -> dict:
    retrieval_keys = sorted({
        key for row in rows for key in row.get("retrieval_metrics", {})
    })
    end_to_end_times = sorted(
        float(row.get("end_to_end_time_ms", 0.0)) for row in rows
    )
    llm_times = sorted(float(row.get("llm_time_ms", 0.0)) for row in rows)

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[round((len(values) - 1) * fraction)]

    resolved_levels: dict[str, int] = {}
    escalation_reasons: dict[str, int] = {}
    for row in rows:
        level = str(row.get("resolved_disclosure_level", "unreported"))
        resolved_levels[level] = resolved_levels.get(level, 0) + 1
        for reason in row.get("escalation_reasons", []):
            reason = str(reason)
            escalation_reasons[reason] = escalation_reasons.get(reason, 0) + 1

    return {
        "completed_tasks": len(rows),
        "api_failures": len(errors),
        "retrieval": {
            key: _mean(rows, lambda row, item=key: float(
                row.get("retrieval_metrics", {}).get(item, 0.0)
            ))
            for key in retrieval_keys
        },
        "agent": {
            "skill_selection_f1": _mean(
                rows, lambda row: float(row["skill_selection_f1"])
            ),
            "sequence_accuracy": _mean(
                rows, lambda row: float(bool(row["sequence_correct"]))
            ),
            "task_success_rate": _mean(
                rows, lambda row: float(row.get("task_success") is True)
            ),
        },
        "efficiency": {
            "avg_skill_context_tokens": _mean(
                rows, lambda row: float(row["skill_context_tokens"])
            ),
            "total_tokens": sum(
                int(row.get("token_usage", {}).get("total_tokens", 0))
                for row in rows
            ),
            "avg_total_tokens_per_task": _mean(
                rows,
                lambda row: float(row.get("token_usage", {}).get("total_tokens", 0)),
            ),
            "avg_skill_calls": _mean(
                rows, lambda row: float(row["skill_calls"])
            ),
            "avg_planner_calls": _mean(
                rows, lambda row: float(row["planner_calls"])
            ),
            "avg_argument_repairs": _mean(
                rows, lambda row: float(row["repair_attempts"])
            ),
            "avg_llm_time_ms": _mean(
                rows, lambda row: float(row.get("llm_time_ms", 0.0))
            ),
            "llm_time_p50_ms": percentile(llm_times, 0.5),
            "llm_time_p95_ms": percentile(llm_times, 0.95),
            "avg_end_to_end_time_ms": _mean(
                rows, lambda row: float(row.get("end_to_end_time_ms", 0.0))
            ),
            "end_to_end_time_p50_ms": percentile(end_to_end_times, 0.5),
            "end_to_end_time_p95_ms": percentile(end_to_end_times, 0.95),
        },
        "disclosure": {
            "resolved_level_counts": resolved_levels,
            "escalation_reason_counts": escalation_reasons,
        },
    }


def _write_checkpoint(path: Path, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data = ROOT / "data" / args.dataset
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    if not task_ids:
        tasks = load_tasks(data / "tasks.jsonl")
        task_ids = [
            task.id for task in tasks
            if args.split == "all" or task.metadata.get("split") == args.split
        ]
    method_names = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in method_names if item not in METHODS]
    if unknown:
        raise ValueError(f"未知方法: {', '.join(unknown)}")
    if args.max_retries < 0:
        raise ValueError("max_retries 不能小于 0")

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "results" / f"task5_planning_{experiment_id}.json"
    )
    if args.resume and output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        for field, expected in {
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "task_ids": task_ids,
        }.items():
            if output.get(field) != expected:
                raise ValueError(f"续跑配置不一致: {field}")
        experiment_id = output["experiment_id"]
    else:
        output = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "task_ids": task_ids,
            "controlled": {
                "retriever": "BM25Retriever(text_level=brief)",
                "candidate_source": "task_fixture",
                "context_budget_source": "task_metadata",
                "temperature": 0.0,
                "thinking": "disabled",
                "initial_environment": "isolated task fixture",
            },
            "methods": {name: METHODS[name] for name in method_names},
            "runs": {},
        }
    _write_checkpoint(output_path, output)

    for method_name in method_names:
        method = METHODS[method_name]
        output["methods"][method_name] = method
        run = output["runs"].setdefault(
            method_name,
            {"per_task": [], "api_errors": [], "summary": {}},
        )
        completed_ids = {row["task_id"] for row in run["per_task"]}
        run["api_errors"] = []
        for task_id in task_ids:
            if task_id in completed_ids:
                continue
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    llm = LLM(
                        provider="deepseek",
                        model=args.model,
                        temperature=0.0,
                        thinking="disabled",
                    )
                    result = run_benchmark(
                        data / "skills.jsonl",
                        data / "tasks.jsonl",
                        retriever=BM25Retriever(text_level="brief"),
                        organizer=HierarchicalOrganizer(method["detail_top_k"]),
                        llm=llm,
                        skill_registry=create_default_skill_registry(),
                        environment_fixtures_path=data / "environment_fixtures.json",
                        task_ids=[task_id],
                        retrieval_ks=(1, 3, 5, 10, 20),
                        max_steps=10,
                        enable_reflection=False,
                        use_task_candidate_fixtures=True,
                        use_task_context_budget=True,
                        planner_mode=method["planner_mode"],
                        max_argument_repairs=method["max_argument_repairs"],
                        run_id=f"task5-{experiment_id}-{method_name}-{task_id}",
                    )
                    row = dict(result["per_task"][0])
                    row["attempts"] = attempt + 1
                    run["per_task"].append(row)
                    print(
                        f"[{method_name}] {task_id} "
                        f"task_success={row.get('task_success')} "
                        f"attempts={attempt + 1}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)
            else:
                run["api_errors"].append({
                    "task_id": task_id,
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "attempts": args.max_retries + 1,
                })
                print(f"[{method_name}] {task_id} API_FAILED", flush=True)
            run["per_task"].sort(key=lambda row: task_ids.index(row["task_id"]))
            run["summary"] = _summary(run["per_task"], run["api_errors"])
            _write_checkpoint(output_path, output)
        print(json.dumps({method_name: run["summary"]}, ensure_ascii=False, indent=2))

    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
