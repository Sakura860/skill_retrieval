"""任务 5：比较 brief、detailed 与 all-field BM25 检索。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_skills, load_tasks  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402
from retrieval.evaluator import aggregate_ranking_metrics  # noqa: E402

LEVELS = ("brief", "detailed", "all")
KS = (1, 3, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument(
        "--dataset",
        default="benchmark_v01",
        choices=("benchmark_v01", "benchmark_v02"),
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def compare(split: str = "dev", dataset: str = "benchmark_v01") -> dict:
    data = ROOT / "data" / dataset
    skills = load_skills(data / "skills.jsonl")
    tasks = load_tasks(data / "tasks.jsonl")
    if split != "all":
        tasks = [task for task in tasks if task.metadata.get("split") == split]

    methods = {}
    for level in LEVELS:
        retriever = BM25Retriever(text_level=level)
        retriever.index(skills)
        rankings = []
        primary_ranks = []
        top_one_hits = 0
        per_task = []
        for task in tasks:
            result = retriever.retrieve(task.instruction, top_k=len(skills))
            ranked_ids = result.ranked_ids()
            gold_ids = list(task.expected_skills)
            rankings.append((ranked_ids, gold_ids))
            top_one_hits += int(bool(ranked_ids) and ranked_ids[0] in set(gold_ids))
            rank = min(
                (ranked_ids.index(skill_id) + 1 for skill_id in gold_ids),
                default=None,
            )
            if rank is not None:
                primary_ranks.append(rank)
            per_task.append({
                "task_id": task.id,
                "gold_skill_ids": gold_ids,
                "ranked_skill_ids": ranked_ids,
                "correct_skill_rank": rank,
            })
        metrics = aggregate_ranking_metrics(rankings, KS)
        methods[level] = {
            "hit@1": top_one_hits / len(tasks) if tasks else 0.0,
            **metrics,
            "mean_correct_skill_rank": (
                sum(primary_ranks) / len(primary_ranks) if primary_ranks else None
            ),
            "per_task": per_task,
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": split,
        "task_count": len(tasks),
        "skill_count": len(skills),
        "controlled_variable": "BM25 indexed Skill fields",
        "methods": methods,
    }


def main() -> None:
    args = parse_args()
    result = compare(args.split, args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
