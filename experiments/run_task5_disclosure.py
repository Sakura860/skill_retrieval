"""Preregistered DeepSeek comparison for brief/schema/full/adaptive disclosure."""
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
from organization.hierarchical import HierarchicalOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "configs" / "task5_preregistered_20260830.json"
CONFIRMATION_REGISTRY = ROOT / "results" / "task5_confirmation_registry.json"
FLOAT_COMPARISON_TOLERANCE = 1e-12
METHODS = {
    "one_stage": {
        "planner_mode": "one_stage",
        "planner_disclosure_level": "full",
        "detail_top_k": 3,
    },
    "always_brief": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "brief",
        "detail_top_k": 0,
    },
    "always_schema": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "schema",
        "detail_top_k": 0,
    },
    "always_full": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "full",
        "detail_top_k": 0,
    },
    "adaptive": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "adaptive",
        "detail_top_k": 0,
    },
    "adaptive_signals": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "adaptive_signals",
        "detail_top_k": 0,
    },
}
SOURCE_FILES = (
    "core/llm.py",
    "core/schemas.py",
    "agent/agent.py",
    "agent/planner.py",
    "agent/disclosure_policy.py",
    "agent/schema_validation.py",
    "evaluation/run_benchmark.py",
    "evaluation/task_metrics.py",
    "evaluation/verifiers.py",
    "execution/handlers.py",
    "organization/hierarchical.py",
    "retrieval/bm25.py",
    "experiments/run_task5_disclosure.py",
    "configs/disclosure_policy_v01.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def reserve_confirmation_run(
    protocol: dict,
    protocol_sha256: str,
    output_path: Path,
    experiment_id: str,
    resume: bool,
    registry_path: Path = CONFIRMATION_REGISTRY,
) -> None:
    """Reserve the protocol's single logical confirmation run.

    An interrupted run still consumes the slot and may only be continued with
    ``--resume`` and the exact same output path. This prevents a failed or
    inconvenient confirmation result from being silently replaced by another.
    """
    policy = protocol.get("confirmation_policy", {})
    maximum_runs = policy.get("maximum_runs")
    if not isinstance(maximum_runs, int) or maximum_runs < 1:
        raise RuntimeError("confirmation_policy.maximum_runs 必须是正整数")
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"runs": []}
    )
    runs = registry.setdefault("runs", [])
    protocol_runs = [
        row for row in runs if row.get("protocol_id") == protocol["protocol_id"]
    ]
    resolved_output = str(output_path.resolve())
    matching = [
        row for row in protocol_runs
        if row.get("output_path") == resolved_output
        and row.get("protocol_sha256") == protocol_sha256
    ]
    if resume:
        if len(matching) != 1 or matching[0].get("status") != "started":
            raise RuntimeError("confirmation 续跑被拒绝：没有匹配的未完成登记")
        return
    if len(protocol_runs) >= maximum_runs:
        raise RuntimeError(
            f"confirmation 运行次数已达上限 {maximum_runs}；禁止再次查看确认集"
        )
    runs.append({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "experiment_id": experiment_id,
        "output_path": resolved_output,
        "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(registry_path, registry)


def complete_confirmation_run(
    protocol: dict,
    protocol_sha256: str,
    output_path: Path,
    registry_path: Path = CONFIRMATION_REGISTRY,
) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    resolved_output = str(output_path.resolve())
    matching = [
        row for row in registry.get("runs", [])
        if row.get("protocol_id") == protocol["protocol_id"]
        and row.get("protocol_sha256") == protocol_sha256
        and row.get("output_path") == resolved_output
    ]
    if len(matching) != 1 or matching[0].get("status") != "started":
        raise RuntimeError("confirmation 完成登记不一致")
    matching[0]["status"] = "completed"
    matching[0]["completed_at"] = datetime.now(timezone.utc).isoformat()
    matching[0]["result_sha256"] = sha256(output_path)
    _write_json(registry_path, registry)


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    required = {"protocol_id", "status", "dataset", "controlled", "decision_rules"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"预注册协议缺少字段: {', '.join(sorted(missing))}")
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_dataset(protocol: dict, data: Path) -> None:
    expected = protocol["dataset"]
    actual = {
        "skills_sha256": sha256(data / "skills.jsonl"),
        "tasks_sha256": sha256(data / "tasks.jsonl"),
        "environment_fixtures_sha256": sha256(data / "environment_fixtures.json"),
    }
    mismatches = [key for key, value in actual.items() if expected.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "数据哈希与预注册协议不一致: " + ", ".join(mismatches)
        )


def source_manifest() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_FILES}


def decision_report(output: dict) -> dict:
    rules = output["protocol_snapshot"]["decision_rules"]
    summaries = {
        name: run.get("summary", {})
        for name, run in output.get("runs", {}).items()
    }

    def values(name: str):
        summary = summaries.get(name, {})
        return (
            summary.get("agent", {}).get("task_success_rate"),
            summary.get("efficiency", {}).get("total_tokens"),
            summary.get("efficiency", {}).get("end_to_end_time_p50_ms"),
        )

    report: dict[str, dict] = {}
    first_rule = rules["two_stage_vs_one_stage"]
    two_stage_method = first_rule.get("candidate_method", "always_full")
    one_success, one_tokens, one_latency = values("one_stage")
    candidate_success, candidate_tokens, candidate_latency = values(two_stage_method)
    if None not in {
        one_success, one_tokens, one_latency,
        candidate_success, candidate_tokens, candidate_latency,
    }:
        checks = {
            "task_success_gain": (
                candidate_success - one_success + FLOAT_COMPARISON_TOLERANCE
                >= first_rule["minimum_absolute_task_success_gain"]
            ),
            "total_token_ratio": candidate_tokens / max(one_tokens, 1)
            <= first_rule["maximum_total_token_ratio"]
            + FLOAT_COMPARISON_TOLERANCE,
            "p50_latency_ratio": candidate_latency / max(one_latency, 1e-9)
            <= first_rule["maximum_p50_end_to_end_latency_ratio"]
            + FLOAT_COMPARISON_TOLERANCE,
        }
        report["two_stage_vs_one_stage"] = {
            "status": "pass" if all(checks.values()) else "fail",
            "candidate_method": two_stage_method,
            "checks": checks,
            "observed": {
                "task_success_gain": candidate_success - one_success,
                "total_token_ratio": candidate_tokens / max(one_tokens, 1),
                "p50_latency_ratio": candidate_latency / max(one_latency, 1e-9),
            },
        }
    else:
        report["two_stage_vs_one_stage"] = {"status": "incomplete"}

    adaptive_method = (
        "adaptive_signals" if "adaptive_signals" in summaries else "adaptive"
    )
    adaptive_success, adaptive_tokens, adaptive_latency = values(adaptive_method)
    full_success, full_tokens, full_latency = values("always_full")
    second_rule = rules["adaptive_vs_always_full"]
    if None not in {
        adaptive_success, adaptive_tokens, adaptive_latency,
        full_success, full_tokens, full_latency,
    }:
        token_reduction = 1 - adaptive_tokens / max(full_tokens, 1)
        latency_reduction = 1 - adaptive_latency / max(full_latency, 1e-9)
        checks = {
            "task_success_drop": full_success - adaptive_success
            <= second_rule["maximum_absolute_task_success_drop"]
            + FLOAT_COMPARISON_TOLERANCE,
            "token_or_latency_reduction": (
                token_reduction + FLOAT_COMPARISON_TOLERANCE
                >= second_rule["minimum_total_token_reduction"]
                or latency_reduction
                + FLOAT_COMPARISON_TOLERANCE
                >= second_rule["minimum_p50_end_to_end_latency_reduction"]
            ),
        }
        report["adaptive_vs_always_full"] = {
            "status": "pass" if all(checks.values()) else "fail",
            "adaptive_method": adaptive_method,
            "checks": checks,
            "observed": {
                "task_success_drop": full_success - adaptive_success,
                "total_token_reduction": token_reduction,
                "p50_latency_reduction": latency_reduction,
            },
        }
    else:
        report["adaptive_vs_always_full"] = {"status": "incomplete"}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", default="dev", choices=("dev", "confirmation"))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise ValueError("max_retries 不能小于 0")
    protocol_path = Path(args.protocol)
    protocol, protocol_sha256 = load_protocol(protocol_path)
    data = ROOT / "data" / protocol["dataset"]["name"]
    validate_dataset(protocol, data)
    if args.split == "confirmation" and protocol["status"] != "frozen":
        raise RuntimeError(
            "confirmation 集被拒绝：协议状态不是 frozen，禁止开发期查看结果"
        )

    allowed_task_ids = (
        protocol["dataset"]["development_task_ids"]
        if args.split == "dev"
        else protocol["dataset"]["confirmation_task_ids"]
    )
    requested_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    task_ids = requested_ids or list(allowed_task_ids)
    unknown_tasks = sorted(set(task_ids) - set(allowed_task_ids))
    if unknown_tasks:
        raise ValueError(
            f"{args.split} 运行包含未授权 task_id: {', '.join(unknown_tasks)}"
        )
    method_names = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown_methods = sorted(set(method_names) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"未知方法: {', '.join(unknown_methods)}")

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else (
        ROOT / "results" / f"task5_disclosure_{args.split}_{experiment_id}.json"
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"输出已存在；如需继续请显式使用 --resume: {output_path}")
    if args.resume and not output_path.exists():
        raise FileNotFoundError(f"--resume 指定的输出不存在: {output_path}")
    if args.split == "confirmation":
        reserve_confirmation_run(
            protocol,
            protocol_sha256,
            output_path,
            experiment_id,
            args.resume,
        )
    if args.resume and output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        for field, expected in {
            "protocol_sha256": protocol_sha256,
            "split": args.split,
            "task_ids": task_ids,
            "source_manifest": source_manifest(),
        }.items():
            if output.get(field) != expected:
                raise ValueError(f"续跑协议不一致: {field}")
        experiment_id = output["experiment_id"]
    else:
        output = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "protocol_snapshot": protocol,
            "source_manifest": source_manifest(),
            "split": args.split,
            "task_ids": task_ids,
            "methods": {name: METHODS[name] for name in method_names},
            "runs": {},
        }
    _write_checkpoint(output_path, output)

    controlled = protocol["controlled"]
    candidate_source = controlled["candidate_source"]
    if candidate_source not in {
        "retriever_top_k",
        "retriever_top_k_gold_augmented_diagnostic",
    }:
        raise ValueError(f"不支持的 candidate_source: {candidate_source}")
    skills = load_skills(data / "skills.jsonl")
    available_task_ids = {task.id for task in load_tasks(data / "tasks.jsonl")}
    missing = sorted(set(task_ids) - available_task_ids)
    if missing:
        raise ValueError(f"数据集中不存在 task_id: {', '.join(missing)}")

    for method_name in method_names:
        method = METHODS[method_name]
        run = output["runs"].setdefault(
            method_name,
            {"per_task": [], "api_errors": [], "summary": {}},
        )
        completed = {row["task_id"] for row in run["per_task"]}
        run["api_errors"] = []
        for task_id in task_ids:
            if task_id in completed:
                continue
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    result = run_benchmark(
                        data / "skills.jsonl",
                        data / "tasks.jsonl",
                        retriever=BM25Retriever(text_level="brief"),
                        organizer=HierarchicalOrganizer(method["detail_top_k"]),
                        llm=LLM(
                            provider="deepseek",
                            model=controlled["model"],
                            temperature=controlled["temperature"],
                            thinking=controlled["thinking"],
                        ),
                        skill_registry=create_default_skill_registry(),
                        environment_fixtures_path=data / "environment_fixtures.json",
                        task_ids=[task_id],
                        top_k=controlled["retrieval_top_k"],
                        retrieval_ks=(1, 3, 5, 10, 20),
                        enable_reflection=False,
                        use_task_candidate_fixtures=False,
                        ensure_gold_in_retrieval=(
                            candidate_source
                            == "retriever_top_k_gold_augmented_diagnostic"
                        ),
                        use_task_context_budget=True,
                        planner_mode=method["planner_mode"],
                        max_argument_repairs=controlled["max_argument_repairs"],
                        planner_disclosure_level=method[
                            "planner_disclosure_level"
                        ],
                        run_id=(
                            f"task5-disclosure-{experiment_id}-{method_name}-{task_id}"
                        ),
                    )
                    row = dict(result["per_task"][0])
                    row["attempts"] = attempt + 1
                    run["per_task"].append(row)
                    print(
                        f"[{method_name}] {task_id} "
                        f"success={row.get('task_success')} "
                        f"level={row.get('resolved_disclosure_level')} "
                        f"tokens={row.get('token_usage', {}).get('total_tokens')} "
                        f"latency_ms={row.get('end_to_end_time_ms'):.1f}",
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
            run["per_task"].sort(key=lambda row: task_ids.index(row["task_id"]))
            run["summary"] = _summary(run["per_task"], run["api_errors"])
            output["decision_report"] = decision_report(output)
            _write_checkpoint(output_path, output)
        print(json.dumps({method_name: run["summary"]}, ensure_ascii=False, indent=2))
    if args.split == "confirmation":
        complete_confirmation_run(protocol, protocol_sha256, output_path)
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
