"""Run internal controls and published SkillRouter baselines under one protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_skills, load_tasks  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402
from retrieval.evaluator import aggregate_ranking_metrics  # noqa: E402
from retrieval.published import (  # noqa: E402
    SkillRouterEmbeddingRetriever,
    SkillRouterRetriever,
)

METHODS = (
    "bm25_brief",
    "bm25_all",
    "skillrouter_embedding",
    "skillrouter_pipeline",
)
KS = (1, 3, 5, 10, 20)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_retriever(method: str, args: argparse.Namespace):
    if method == "bm25_brief":
        return BM25Retriever(text_level="brief")
    if method == "bm25_all":
        return BM25Retriever(text_level="all")
    common = {
        "max_length": args.encoder_max_length,
        "batch_size": args.encoder_batch_size,
        "device": args.device,
        "dtype": args.dtype,
    }
    if method == "skillrouter_embedding":
        return SkillRouterEmbeddingRetriever(**common)
    if method == "skillrouter_pipeline":
        return SkillRouterRetriever(
            **common,
            retrieval_top_k=args.retrieval_top_k,
            reranker_max_length=args.reranker_max_length,
            reranker_batch_size=args.reranker_batch_size,
            prompt_format=args.prompt_format,
        )
    raise ValueError(f"未知方法: {method}")


def retriever_provenance(retriever) -> dict:
    if hasattr(retriever, "provenance"):
        return retriever.provenance()
    return {
        "method": type(retriever).__name__,
        "implementation": "skill-agent local control",
        "text_level": getattr(retriever, "text_level", None),
        "fallback_allowed": False,
    }


def run_method(retriever, skills, tasks) -> dict:
    prepare_started = time.perf_counter()
    if hasattr(retriever, "prepare"):
        retriever.prepare()
    model_prepare_ms = (time.perf_counter() - prepare_started) * 1000

    started = time.perf_counter()
    retriever.index(skills)
    index_ms = (time.perf_counter() - started) * 1000
    rankings = []
    per_task = []
    query_times = []
    for task in tasks:
        query_started = time.perf_counter()
        result = retriever.retrieve(task.instruction, top_k=len(skills))
        query_ms = (time.perf_counter() - query_started) * 1000
        query_times.append(query_ms)
        ranked_ids = result.ranked_ids()
        gold_ids = list(task.expected_skills)
        rankings.append((ranked_ids, gold_ids))
        per_task.append({
            "task_id": task.id,
            "gold_skill_ids": gold_ids,
            "ranked_skill_ids": ranked_ids,
            "scores": [float(score) for score in result.scores],
            "query_ms": query_ms,
        })
    sorted_times = sorted(query_times)

    def percentile(fraction: float) -> float:
        if not sorted_times:
            return 0.0
        index = round((len(sorted_times) - 1) * fraction)
        return float(sorted_times[index])

    return {
        "status": "completed",
        "provenance": retriever_provenance(retriever),
        "metrics": aggregate_ranking_metrics(rankings, KS),
        "latency_ms": {
            "model_prepare": model_prepare_ms,
            "index": index_ms,
            "query_mean": sum(query_times) / len(query_times) if query_times else 0.0,
            "query_p50": percentile(0.5),
            "query_p95": percentile(0.95),
        },
        "per_task": per_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark_v02")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["bm25_brief", "bm25_all"],
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--encoder-max-length", type=int, default=4096)
    parser.add_argument("--reranker-max-length", type=int, default=4096)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument(
        "--prompt-format",
        choices=("flat-full", "flat-nd", "struct"),
        default="flat-full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "data" / args.dataset
    skills_path = data_dir / "skills.jsonl"
    tasks_path = data_dir / "tasks.jsonl"
    skills = load_skills(skills_path)
    tasks = load_tasks(tasks_path)
    if args.split != "all":
        tasks = [task for task in tasks if task.metadata.get("split") == args.split]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "published_baselines_v02",
        "dataset": args.dataset,
        "split": args.split,
        "task_count": len(tasks),
        "skill_count": len(skills),
        "dataset_sha256": {
            "skills": file_sha256(skills_path),
            "tasks": file_sha256(tasks_path),
        },
        "methods": {},
    }
    try:
        for method in args.methods:
            result["methods"][method] = run_method(
                build_retriever(method, args), skills, tasks
            )
    finally:
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
