"""Matched dev diagnostic for typed completion and official GoS reverse-PPR."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from data.loader import load_skills, load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from experiments.run_task5_planning import _summary, _write_checkpoint  # noqa: E402
from organization.gos_official import (  # noqa: E402
    OfficialGoSReversePPROrganizer,
    load_official_runtime,
)
from organization.graph import GraphOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "configs" / "gos_structural_protocol_20260903.json"
DEFAULT_UPSTREAM = ROOT.parent / "tmp" / "graph-of-skills-upstream-20260903"
DEFAULT_OUTPUT = ROOT / "results" / "gos_structural_baseline_dev_20260903.json"
SOURCE_FILES = (
    "organization/typed_graph.py",
    "organization/graph.py",
    "organization/gos_official.py",
    "experiments/run_gos_structural_baseline.py",
    "evaluation/run_benchmark.py",
    "evaluation/task_metrics.py",
    "agent/agent.py",
    "agent/planner.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_verify(
    protocol_path: Path,
    upstream_root: Path,
) -> tuple[dict[str, Any], str, Path]:
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if protocol.get("status") != "frozen_dev_diagnostic":
        raise RuntimeError("GoS structural protocol must be frozen before execution")
    data = ROOT / "data" / protocol["scope"]["dataset"]
    for filename, expected in protocol["dataset_hashes"].items():
        actual = sha256(data / filename)
        if actual != expected:
            raise RuntimeError(f"Dataset hash mismatch: {filename}")
    official = protocol["official_graph_of_skills"]
    runtime = load_official_runtime(upstream_root)
    if not callable(runtime.personalized_pagerank):
        raise RuntimeError("Pinned official runtime lacks personalized_pagerank")
    query_path = upstream_root / official["runtime_path"]
    if sha256(query_path) != official["runtime_sha256"]:
        raise RuntimeError("Official Graph-of-Skills runtime hash mismatch")
    return protocol, hashlib.sha256(raw).hexdigest(), data


def source_manifest() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_FILES}


def organizer_for(
    method: str,
    skills,
    upstream_root: Path,
    maximum: int,
):
    if method == "no_completion":
        return GraphOrganizer(catalog=skills, max_additional_skills=0)
    if method == "typed_prerequisite_completion":
        return GraphOrganizer(catalog=skills, max_additional_skills=maximum)
    if method == "gos_reverse_ppr_prerequisite":
        return OfficialGoSReversePPROrganizer(
            skills,
            upstream_root,
            max_additional_skills=maximum,
            include_dataflow=False,
        )
    if method == "gos_reverse_ppr_prerequisite_plus_dataflow":
        return OfficialGoSReversePPROrganizer(
            skills,
            upstream_root,
            max_additional_skills=maximum,
            include_dataflow=True,
        )
    raise ValueError(f"Unknown structural method: {method}")


def structural_row(row: dict[str, Any], task) -> dict[str, Any]:
    required = set(task.metadata.get("intentionally_omitted_skill_ids", []))
    added = list(row.get("added_skill_ids", []))
    added_set = set(added)
    correct = sorted(required & added_set)
    irrelevant = sorted(added_set - required)
    return {
        "task_id": task.id,
        "required_missing_ids": sorted(required),
        "added_skill_ids": added,
        "correct_added_skill_ids": correct,
        "irrelevant_added_skill_ids": irrelevant,
        "missing_prerequisite_recall": (
            len(correct) / len(required) if required else 1.0
        ),
        "candidate_growth": len(added),
        "exposed_skill_count": len(row.get("exposed_skill_ids", [])),
        "skill_context_tokens": int(row.get("skill_context_tokens", 0)),
        "total_tokens": int(row.get("token_usage", {}).get("total_tokens", 0)),
        "end_to_end_time_ms": float(row.get("end_to_end_time_ms", 0)),
        "task_success": row.get("task_success"),
        "selection_f1": float(row.get("skill_selection_f1", 0)),
        "failure_reason": row.get("failure_reason"),
        "graph_issues": list(row.get("graph_issues", [])),
    }


def aggregate(output: dict[str, Any]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for method, method_run in output["runs"].items():
        structural = [
            row
            for repetition in method_run.get("repetitions", [])
            for row in repetition.get("structural", [])
        ]
        summaries = [
            repetition.get("summary", {})
            for repetition in method_run.get("repetitions", [])
        ]
        api_errors = sum(
            len(repetition.get("api_errors", []))
            for repetition in method_run.get("repetitions", [])
        )
        aggregates[method] = {
            "observation_count": len(structural),
            "independent_task_count": len({row["task_id"] for row in structural}),
            "api_errors": api_errors,
            "mean_missing_prerequisite_recall": statistics.fmean(
                row["missing_prerequisite_recall"] for row in structural
            ) if structural else 0.0,
            "mean_correct_additions": statistics.fmean(
                len(row["correct_added_skill_ids"]) for row in structural
            ) if structural else 0.0,
            "mean_irrelevant_additions": statistics.fmean(
                len(row["irrelevant_added_skill_ids"]) for row in structural
            ) if structural else 0.0,
            "mean_candidate_growth": statistics.fmean(
                row["candidate_growth"] for row in structural
            ) if structural else 0.0,
            "mean_context_tokens": statistics.fmean(
                row["skill_context_tokens"] for row in structural
            ) if structural else 0.0,
            "total_llm_tokens": sum(row["total_tokens"] for row in structural),
            "task_success_rate": statistics.fmean(
                bool(row["task_success"]) for row in structural
            ) if structural else 0.0,
            "mean_end_to_end_time_ms": statistics.fmean(
                row["end_to_end_time_ms"] for row in structural
            ) if structural else 0.0,
            "all_repetition_summaries": summaries,
        }
    return aggregates


def render_report(output: dict[str, Any]) -> str:
    lines = [
        "# Graph-of-Skills matched structural baseline",
        "",
        "This dev-only diagnostic uses the pinned official reverse-aware PPR "
        "runtime with fixed task candidate seeds. It is not the full hybrid GoS pipeline.",
        "",
        f"- Protocol: `{output['protocol_snapshot']['protocol_id']}`",
        f"- Official upstream: `{output['official_provenance']['upstream_commit']}`",
        f"- Independent tasks: {output['protocol_snapshot']['scope']['independent_tasks']}; "
        f"model repetitions: {output['protocol_snapshot']['scope']['model_repetitions']}",
        "- Same task candidates, graph edges, maximum two additions, context budget, model, and Planner settings for every method.",
        "",
        "| Method | Prereq recall | Correct add | Irrelevant add | Growth | Context tokens | LLM tokens | Task success | Mean e2e ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in output["aggregates"].items():
        lines.append(
            f"| {method} | {values['mean_missing_prerequisite_recall']:.3f} | "
            f"{values['mean_correct_additions']:.2f} | "
            f"{values['mean_irrelevant_additions']:.2f} | "
            f"{values['mean_candidate_growth']:.2f} | "
            f"{values['mean_context_tokens']:.1f} | "
            f"{values['total_llm_tokens']} | "
            f"{values['task_success_rate']:.3f} | "
            f"{values['mean_end_to_end_time_ms']:.2f} |"
        )
    lines.extend([
        "",
        "## Per-task structural outputs",
        "",
        "The added IDs below are deterministic and identical across the three model repetitions; task success is listed as successes/3.",
        "",
        "| Method | Task | Added | Correct | Irrelevant | Successes/3 |",
        "|---|---|---|---|---|---:|",
    ])
    for method, method_run in output["runs"].items():
        task_ids = output["task_ids"]
        for task_id in task_ids:
            rows = [
                row
                for repetition in method_run["repetitions"]
                for row in repetition["structural"]
                if row["task_id"] == task_id
            ]
            first = rows[0]
            lines.append(
                f"| {method} | {task_id} | "
                f"{','.join(first['added_skill_ids']) or '-'} | "
                f"{','.join(first['correct_added_skill_ids']) or '-'} | "
                f"{','.join(first['irrelevant_added_skill_ids']) or '-'} | "
                f"{sum(row['task_success'] is True for row in rows)}/3 |"
            )
    dep = output["aggregates"].get("gos_reverse_ppr_prerequisite", {})
    flow = output["aggregates"].get(
        "gos_reverse_ppr_prerequisite_plus_dataflow", {}
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Exact typed completion follows only prerequisite edges and stops after the required closure; it should therefore be read as a high-precision local operator, not a general graph ranker.",
        "- Official reverse-PPR can recover a prerequisite through reverse dependency propagation, but a fixed top-N bundle may also hydrate unrelated zero/low-score nodes when the graph is sparse. The irrelevant-addition metric makes that cost explicit.",
        (
            "- Adding dataflow as GoS `workflow` edges produced no aggregate change on these two tasks. Each dataflow edge is collinear with its prerequisite edge, so this dataset provides no independent evidence that dataflow improves success."
            if dep and flow and all(
                dep.get(key) == flow.get(key)
                for key in (
                    "mean_missing_prerequisite_recall",
                    "mean_correct_additions",
                    "mean_irrelevant_additions",
                    "task_success_rate",
                )
            )
            else "- Prerequisite-only and prerequisite-plus-dataflow results differ; inspect the per-task rows before attributing the change to dataflow."
        ),
        "- `co_use`, `alternative`, and `conflict` are not tested here and receive no effectiveness claim.",
        "",
        "## Reproduction and validity boundary",
        "",
        "- The adapter imports the exact pinned official `query.py` after checking both git commit and SHA-256; it does not copy or reimplement PPR.",
        "- Full GoS hybrid semantic/lexical seeding was not run because no compatible embedding credential/workspace is configured. This missing result is not replaced with Hash embeddings.",
        "- Only two independent dev tasks are used. Three DeepSeek repetitions assess runtime stability but are not six independent tasks; no held-out or broad superiority claim is made.",
        "- No consumed Task 5 confirmation task or result was used to tune or run this baseline.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--upstream-root", default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    upstream_root = Path(args.upstream_root).resolve()
    output_path = Path(args.output)
    protocol, protocol_hash, data = load_and_verify(protocol_path, upstream_root)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output_path}")
    skills = load_skills(data / "skills.jsonl")
    all_tasks = {task.id: task for task in load_tasks(data / "tasks.jsonl")}
    task_ids = list(protocol["scope"]["task_ids"])
    tasks = {task_id: all_tasks[task_id] for task_id in task_ids}
    controlled = protocol["controlled"]
    methods = list(controlled["methods"])
    repeat_count = int(protocol["scope"]["model_repetitions"])

    official_organizer = OfficialGoSReversePPROrganizer(
        skills,
        upstream_root,
        max_additional_skills=controlled["max_additional_skills"],
    )
    if args.resume:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        for key, expected in {
            "protocol_sha256": protocol_hash,
            "source_manifest": source_manifest(),
            "task_ids": task_ids,
        }.items():
            if output.get(key) != expected:
                raise RuntimeError(f"Resume mismatch: {key}")
    else:
        output = {
            "experiment_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": protocol_hash,
            "protocol_snapshot": protocol,
            "source_manifest": source_manifest(),
            "official_provenance": official_organizer.provenance(),
            "task_ids": task_ids,
            "runs": {method: {"repetitions": []} for method in methods},
        }
        _write_checkpoint(output_path, output)

    for method in methods:
        method_run = output["runs"][method]
        while len(method_run["repetitions"]) < repeat_count:
            repeat_index = len(method_run["repetitions"]) + 1
            repetition = {
                "repeat": repeat_index,
                "per_task": [],
                "structural": [],
                "api_errors": [],
                "summary": {},
            }
            for task_id in task_ids:
                last_error: Exception | None = None
                for attempt in range(args.max_retries + 1):
                    try:
                        result = run_benchmark(
                            data / "skills.jsonl",
                            data / "tasks.jsonl",
                            retriever=BM25Retriever(text_level="brief"),
                            organizer=organizer_for(
                                method,
                                skills,
                                upstream_root,
                                controlled["max_additional_skills"],
                            ),
                            llm=LLM(
                                provider="deepseek",
                                model=controlled["model"],
                                temperature=controlled["temperature"],
                                thinking=controlled["thinking"],
                            ),
                            skill_registry=create_default_skill_registry(),
                            environment_fixtures_path=data / "environment_fixtures.json",
                            task_ids=[task_id],
                            top_k=len(tasks[task_id].metadata["candidate_skill_ids"]),
                            retrieval_ks=(1, 5, 10, 24),
                            enable_reflection=False,
                            use_task_candidate_fixtures=True,
                            use_task_context_budget=True,
                            planner_mode=controlled["planner_mode"],
                            max_argument_repairs=controlled["max_argument_repairs"],
                            planner_disclosure_level="full",
                            run_id=f"gos-{output['experiment_id']}-{method}-r{repeat_index}-{task_id}",
                        )
                        row = dict(result["per_task"][0])
                        row["attempts"] = attempt + 1
                        repetition["per_task"].append(row)
                        repetition["structural"].append(
                            structural_row(row, tasks[task_id])
                        )
                        print(
                            f"[{method} r{repeat_index}] {task_id} "
                            f"added={row.get('added_skill_ids')} "
                            f"success={row.get('task_success')} "
                            f"tokens={row.get('token_usage', {}).get('total_tokens')}",
                            flush=True,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.max_retries:
                            time.sleep(2 ** attempt)
                else:
                    repetition["api_errors"].append({
                        "task_id": task_id,
                        "error": f"{type(last_error).__name__}: {last_error}",
                        "attempts": args.max_retries + 1,
                    })
            repetition["summary"] = _summary(
                repetition["per_task"], repetition["api_errors"]
            )
            method_run["repetitions"].append(repetition)
            output["aggregates"] = aggregate(output)
            _write_checkpoint(output_path, output)

    output["aggregates"] = aggregate(output)
    _write_checkpoint(output_path, output)
    report_path = output_path.with_suffix(".md")
    report_path.write_text(render_report(output), encoding="utf-8")
    print(f"Result saved: {output_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
