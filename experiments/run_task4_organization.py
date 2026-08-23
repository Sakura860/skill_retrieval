"""任务 4：固定 BM25 候选与预算的 Flat/Hierarchical 公平实验。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from data.loader import load_skills, load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from organization.flat import FlatOrganizer  # noqa: E402
from organization.graph import GraphOrganizer  # noqa: E402
from organization.hierarchical import HierarchicalOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DATA = ROOT / "data" / "benchmark_v01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--detail-top-k", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_one(
    task_id: str,
    organizer: Any,
    model: str,
    repeat: int,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            llm = LLM(
                provider="deepseek",
                model=model,
                temperature=0.0,
                thinking="disabled",
            )
            result = run_benchmark(
                DATA / "skills.jsonl",
                DATA / "tasks.jsonl",
                retriever=BM25Retriever(text_level="brief"),
                organizer=organizer,
                llm=llm,
                skill_registry=create_default_skill_registry(),
                environment_fixtures_path=DATA / "environment_fixtures.json",
                task_ids=[task_id],
                retrieval_ks=(1, 3, 5, 10, 20),
                max_steps=10,
                enable_reflection=False,
                use_task_candidate_fixtures=True,
                use_task_context_budget=True,
                run_id=f"task4-{type(organizer).__name__}-r{repeat}-{task_id}",
            )
            row = dict(result["per_task"][0])
            row.update({
                "repeat": repeat,
                "organizer": type(organizer).__name__,
                "api_error": None,
                "attempts": attempt + 1,
            })
            return row
        except Exception as exc:  # 单条失败必须保留，并允许后续任务继续
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return {
        "task_id": task_id,
        "repeat": repeat,
        "organizer": type(organizer).__name__,
        "task_success": False,
        "skill_selection_f1": 0.0,
        "sequence_correct": False,
        "skill_calls": 0,
        "redundant_calls": 0,
        "skill_context_tokens": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "failure_reason": "api_error",
        "api_error": f"{type(last_error).__name__}: {last_error}",
        "attempts": max_retries + 1,
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    calls = sum(int(row.get("skill_calls", 0)) for row in rows)
    redundant = sum(int(row.get("redundant_calls", 0)) for row in rows)
    return {
        "observations": count,
        "api_failures": sum(bool(row.get("api_error")) for row in rows),
        "task_success_rate": (
            sum(row.get("task_success") is True for row in rows) / count if count else None
        ),
        "skill_selection_f1": (
            sum(float(row.get("skill_selection_f1", 0.0)) for row in rows) / count
            if count else None
        ),
        "sequence_accuracy": (
            sum(bool(row.get("sequence_correct", False)) for row in rows) / count
            if count else None
        ),
        "avg_context_tokens": (
            sum(int(row.get("skill_context_tokens", 0)) for row in rows) / count
            if count else None
        ),
        "total_tokens": sum(
            int(row.get("token_usage", {}).get("total_tokens", 0)) for row in rows
        ),
        "redundant_call_rate": redundant / calls if calls else 0.0,
    }


def _repeat_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repeat[int(row["repeat"])].append(row)
    summaries = [_metric_summary(items) for _, items in sorted(by_repeat.items())]
    output: dict[str, dict[str, float]] = {}
    for metric in (
        "task_success_rate",
        "skill_selection_f1",
        "sequence_accuracy",
        "avg_context_tokens",
        "total_tokens",
        "redundant_call_rate",
    ):
        values = [float(item[metric]) for item in summaries if item[metric] is not None]
        output[metric] = {
            "mean": statistics.mean(values) if values else math.nan,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return output


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    stratified: list[dict[str, Any]] = []
    for organizer in sorted({row["organizer"] for row in rows}):
        method_rows = [row for row in rows if row["organizer"] == organizer]
        overall[organizer] = {
            **_metric_summary(method_rows),
            "across_repeat": _repeat_stats(method_rows),
        }
        for field in ("candidate_count", "target_gold_rank", "context_budget_tokens"):
            values = sorted({row.get(field) for row in method_rows if row.get(field) is not None})
            for value in values:
                group = [row for row in method_rows if row.get(field) == value]
                stratified.append({
                    "organizer": organizer,
                    "dimension": field,
                    "value": value,
                    **_metric_summary(group),
                })
    return {"overall": overall, "stratified": stratified}


def _graph_diagnostics() -> dict[str, Any]:
    skills = load_skills(DATA / "graph_skills.jsonl")
    tasks = load_tasks(DATA / "graph_tasks.jsonl")
    by_id = {skill.id: skill for skill in skills}
    cases = []
    for task in tasks:
        candidate_ids = list(task.metadata["candidate_skill_ids"])
        candidates = [by_id[item] for item in candidate_ids]
        graph = GraphOrganizer().organize_context(candidates, task)
        positions = {item: index for index, item in enumerate(graph.detailed_skill_ids)}
        violations = []
        for skill_id in graph.detailed_skill_ids:
            for dependency in by_id[skill_id].dependencies:
                if dependency in positions and positions[dependency] > positions[skill_id]:
                    violations.append([dependency, skill_id])
        cases.append({
            "task_id": task.id,
            "diagnostic_type": task.metadata["diagnostic_type"],
            "flat_candidate_order": candidate_ids,
            "graph_order": graph.detailed_skill_ids,
            "dependency_order_violations": violations,
            "missing_dependency_warning": "缺少依赖" in graph.text,
            "cycle_warning": "依赖环" in graph.text,
            "expected_graph_issue": task.metadata.get("expected_graph_issue"),
        })
    return {
        "scope": "structural diagnostics only; no automatic dependency completion",
        "task_count": len(cases),
        "dependency_clean_cases": sum(not case["dependency_order_violations"] for case in cases),
        "missing_dependency_warnings": sum(case["missing_dependency_warning"] for case in cases),
        "cycle_warnings": sum(case["cycle_warning"] for case in cases),
        "cases": cases,
    }


def _failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["repeat"]), row["task_id"])][row["organizer"]] = row
    cases = []
    for (repeat, task_id), methods in paired.items():
        if len(methods) != 2:
            continue
        flat = methods.get("FlatOrganizer", {})
        hierarchical = methods.get("HierarchicalOrganizer", {})
        disagreement = flat.get("task_success") != hierarchical.get("task_success")
        context_delta = int(flat.get("skill_context_tokens", 0)) - int(
            hierarchical.get("skill_context_tokens", 0)
        )
        if disagreement:
            cases.append({
                "case_type": "outcome_disagreement",
                "repeat": repeat,
                "task_id": task_id,
                "flat": flat,
                "hierarchical": hierarchical,
                "context_token_saving": context_delta,
            })
    if len(cases) < 3:
        representatives = []
        for (repeat, task_id), methods in paired.items():
            if len(methods) != 2:
                continue
            flat = methods["FlatOrganizer"]
            hierarchical = methods["HierarchicalOrganizer"]
            representatives.append((
                int(flat.get("skill_context_tokens", 0))
                - int(hierarchical.get("skill_context_tokens", 0)),
                repeat,
                task_id,
                flat,
                hierarchical,
            ))
        for delta, repeat, task_id, flat, hierarchical in sorted(
            representatives,
            reverse=True,
        ):
            if any(case["repeat"] == repeat and case["task_id"] == task_id for case in cases):
                continue
            cases.append({
                "case_type": "largest_context_contrast",
                "repeat": repeat,
                "task_id": task_id,
                "flat": flat,
                "hierarchical": hierarchical,
                "context_token_saving": delta,
            })
            if len(cases) >= 3:
                break
    return cases


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "organizer", "dimension", "value", "observations", "api_failures",
        "task_success_rate", "skill_selection_f1", "sequence_accuracy",
        "avg_context_tokens", "total_tokens", "redundant_call_rate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("正式实验 repeats 必须至少为 3")
    selected_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    tasks = load_tasks(DATA / "tasks.jsonl")
    if args.split != "all":
        tasks = [task for task in tasks if task.metadata.get("split") == args.split]
    if selected_ids:
        tasks = [task for task in tasks if task.id in selected_ids]
    if not tasks:
        raise ValueError("没有匹配的任务")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "results" / f"task4_organization_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment_id": output_dir.name,
        "timestamp_utc": timestamp,
        "hypothesis": (
            "With identical BM25 candidates and context budgets, hierarchical "
            "brief-plus-top-detail disclosure reduces context without lowering task success."
        ),
        "split": args.split,
        "task_ids": [task.id for task in tasks],
        "task_count": len(tasks),
        "repeats": args.repeats,
        "retriever": "BM25Retriever(text_level=brief)",
        "candidate_source": "task.metadata.candidate_skill_ids",
        "context_budget_source": "task.metadata.slice.context_budget_tokens",
        "organizers": ["FlatOrganizer", "HierarchicalOrganizer"],
        "hierarchical_detail_top_k": args.detail_top_k,
        "model": args.model,
        "temperature": 0.0,
        "thinking": "disabled",
        "reflection": False,
        "initial_state": "fresh isolated TaskEnvironment per observation",
    }
    _write_json(output_dir / "config_snapshot.json", config)
    _write_json(output_dir / "graph_diagnostics.json", _graph_diagnostics())

    rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "per_task.jsonl"
    organizers = (
        FlatOrganizer(),
        HierarchicalOrganizer(detail_top_k=args.detail_top_k),
    )
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for repeat in range(1, args.repeats + 1):
            for task in tasks:
                ordered = organizers if (repeat + len(rows)) % 2 else organizers[::-1]
                for organizer in ordered:
                    row = _run_one(
                        task.id,
                        organizer,
                        args.model,
                        repeat,
                        args.max_retries,
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        f"[{len(rows)}/{len(tasks) * args.repeats * 2}] "
                        f"r{repeat} {task.id} {type(organizer).__name__} "
                        f"success={row.get('task_success')} "
                        f"tokens={row.get('token_usage', {}).get('total_tokens', 0)}",
                        flush=True,
                    )

    aggregated = aggregate(rows)
    _write_json(output_dir / "aggregate.json", aggregated)
    csv_rows = []
    for organizer, summary in aggregated["overall"].items():
        csv_rows.append({
            "organizer": organizer,
            "dimension": "overall",
            "value": "all",
            **{key: value for key, value in summary.items() if key != "across_repeat"},
        })
    csv_rows.extend(aggregated["stratified"])
    _write_csv(output_dir / "aggregate.csv", csv_rows)
    failures = {
        "failure_type_counts": dict(Counter(
            row.get("failure_reason") or "success"
            for row in rows
        )),
        "cases": _failure_cases(rows),
    }
    _write_json(output_dir / "failure_cases.json", failures)
    print(output_dir)
    print(json.dumps(aggregated["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
