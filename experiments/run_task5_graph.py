"""任务 5：typed graph prerequisite completion 的独立 DeepSeek 诊断。"""
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
from data.loader import load_skills, load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from organization.graph import GraphOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402
from experiments.run_task5_planning import _summary, _write_checkpoint  # noqa: E402

DATA = ROOT / "data" / "benchmark_v02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills = load_skills(DATA / "skills.jsonl")
    graph_task_ids = [
        task.id for task in load_tasks(DATA / "tasks.jsonl")
        if task.metadata.get("split") == "dev"
        and "prerequisite_completion" in task.metadata.get("research_questions", [])
    ]
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "results" / f"task5_graph_{experiment_id}.json"
    )
    output = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "benchmark_v02",
        "split": "dev",
        "task_ids": graph_task_ids,
        "model": args.model,
        "controlled": {
            "original_candidates": "task fixture with prerequisite intentionally omitted",
            "typed_edges": "task.metadata.graph_edges",
            "only_changed_variable": "max_additional_skills: 0 versus 2",
            "temperature": 0.0,
            "thinking": "disabled",
        },
        "runs": {},
    }
    _write_checkpoint(output_path, output)
    for method_name, max_additional in (
        ("without_completion", 0),
        ("with_prerequisite_completion", 2),
    ):
        run = {"per_task": [], "api_errors": [], "summary": {}}
        output["runs"][method_name] = run
        for task_id in graph_task_ids:
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    result = run_benchmark(
                        DATA / "skills.jsonl",
                        DATA / "tasks.jsonl",
                        retriever=BM25Retriever(text_level="brief"),
                        organizer=GraphOrganizer(
                            catalog=skills,
                            max_additional_skills=max_additional,
                        ),
                        llm=LLM(
                            provider="deepseek",
                            model=args.model,
                            temperature=0.0,
                            thinking="disabled",
                        ),
                        skill_registry=create_default_skill_registry(),
                        environment_fixtures_path=DATA / "environment_fixtures.json",
                        task_ids=[task_id],
                        retrieval_ks=(1, 5, 10, 24),
                        enable_reflection=False,
                        use_task_candidate_fixtures=True,
                        use_task_context_budget=True,
                        planner_mode="one_stage",
                        max_argument_repairs=0,
                        run_id=f"task5-graph-{experiment_id}-{method_name}-{task_id}",
                    )
                    row = dict(result["per_task"][0])
                    row["attempts"] = attempt + 1
                    run["per_task"].append(row)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)
            else:
                run["api_errors"].append({
                    "task_id": task_id,
                    "error": f"{type(last_error).__name__}: {last_error}",
                })
            run["summary"] = _summary(run["per_task"], run["api_errors"])
            _write_checkpoint(output_path, output)
        print(json.dumps({
            method_name: {
                "summary": run["summary"],
                "added_skill_ids": {
                    row["task_id"]: row["added_skill_ids"]
                    for row in run["per_task"]
                },
            }
        }, ensure_ascii=False, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
