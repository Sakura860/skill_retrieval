"""运行 Task 5 最终组合候选；配置冻结前强制拒绝 held-out test。"""
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

from core.llm import LLM  # noqa: E402
from data.loader import load_skills, load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from experiments.run_task5_planning import _summary, _write_checkpoint  # noqa: E402
from organization.graph import GraphOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "task5_candidate_20260824.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--output", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    required = {"config_id", "status", "model", "retrieval", "organization", "agent", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"候选配置缺少字段: {', '.join(sorted(missing))}")
    if config["retrieval"].get("candidate_source") != "retriever_top_k":
        raise ValueError("最终组合必须使用真实 retriever top-k，而不是任务候选 fixture")
    if config["organization"].get("expand_relation_types") != ["prerequisite"]:
        raise ValueError("typed graph 只允许 prerequisite 自动扩张")
    return config, hashlib.sha256(raw).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise ValueError("max_retries 不能小于 0")
    config_path = Path(args.config)
    config, config_sha256 = _load_config(config_path)
    if args.split == "test" and config["status"] != "frozen":
        raise RuntimeError("held-out test 被拒绝：请先完成 dev 重复并将配置状态冻结")

    dataset = config["evaluation"]["dataset"]
    data = ROOT / "data" / dataset
    skills = load_skills(data / "skills.jsonl")
    tasks = [
        task for task in load_tasks(data / "tasks.jsonl")
        if task.metadata.get("split") == args.split
    ]
    task_ids = [task.id for task in tasks]
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else (
        ROOT / "results" / f"task5_candidate_{args.split}_{experiment_id}.json"
    )
    if args.resume and output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        for field, expected in {
            "config_sha256": config_sha256,
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
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "config_snapshot": config,
            "dataset": dataset,
            "split": args.split,
            "task_ids": task_ids,
            "per_task": [],
            "api_errors": [],
            "summary": {},
        }
    _write_checkpoint(output_path, output)

    completed_ids = {row["task_id"] for row in output["per_task"]}
    output["api_errors"] = []
    for task_id in task_ids:
        if task_id in completed_ids:
            continue
        last_error = None
        for attempt in range(args.max_retries + 1):
            try:
                result = run_benchmark(
                    data / "skills.jsonl",
                    data / "tasks.jsonl",
                    retriever=BM25Retriever(
                        text_level=config["retrieval"]["text_level"]
                    ),
                    organizer=GraphOrganizer(
                        catalog=skills,
                        max_additional_skills=config["organization"]["max_additional_skills"],
                    ),
                    llm=LLM(
                        provider=config["model"]["provider"],
                        model=config["model"]["name"],
                        temperature=config["model"]["temperature"],
                        thinking=config["model"]["thinking"],
                    ),
                    skill_registry=create_default_skill_registry(),
                    environment_fixtures_path=data / "environment_fixtures.json",
                    task_ids=[task_id],
                    top_k=config["retrieval"]["top_k"],
                    retrieval_ks=tuple(config["evaluation"]["retrieval_ks"]),
                    max_steps=config["agent"]["max_steps"],
                    enable_reflection=config["agent"]["enable_reflection"],
                    use_task_candidate_fixtures=False,
                    use_task_context_budget=True,
                    planner_mode=config["agent"]["planner_mode"],
                    max_argument_repairs=config["agent"]["max_argument_repairs"],
                    run_id=f"task5-candidate-{experiment_id}-{args.split}-{task_id}",
                )
                row = dict(result["per_task"][0])
                row["attempts"] = attempt + 1
                output["per_task"].append(row)
                print(
                    f"[{args.split}] {task_id} task_success={row.get('task_success')} "
                    f"repairs={row.get('repair_attempts')} attempts={attempt + 1}",
                    flush=True,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
        else:
            output["api_errors"].append({
                "task_id": task_id,
                "error": f"{type(last_error).__name__}: {last_error}",
                "attempts": args.max_retries + 1,
            })
        output["per_task"].sort(key=lambda row: task_ids.index(row["task_id"]))
        output["summary"] = _summary(output["per_task"], output["api_errors"])
        _write_checkpoint(output_path, output)

    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
