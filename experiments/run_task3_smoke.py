"""用真实 DeepSeek 对 task3 的四类单步 dev 任务做小规模验证。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from organization.flat import FlatOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DATA = ROOT / "data" / "benchmark_v01"
DEFAULT_TASK_IDS = ["tcalc01", "tjson01", "tfile01", "tdb01"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--task-ids",
        default=",".join(DEFAULT_TASK_IDS),
        help="逗号分隔的 task id",
    )
    parser.add_argument("--top-k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    if not task_ids:
        raise ValueError("至少提供一个 task id")

    llm = LLM(
        provider="deepseek",
        model=args.model,
        temperature=0.0,
        thinking="disabled",
    )
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=FlatOrganizer(),
        llm=llm,
        skill_registry=create_default_skill_registry(),
        environment_fixtures_path=DATA / "environment_fixtures.json",
        task_ids=task_ids,
        top_k=args.top_k,
        retrieval_ks=(1, 5, 10, 24),
        enable_reflection=False,
        run_id="task3-deepseek-smoke",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"task3_deepseek_smoke_{timestamp}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(result["metrics"]["agent"], ensure_ascii=False))


if __name__ == "__main__":
    main()
