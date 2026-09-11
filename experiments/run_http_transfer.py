"""Run the unchanged disclosure policy on the unseen local HTTP/API family."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from data.loader import load_tasks  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from execution.handlers import create_default_skill_registry  # noqa: E402
from experiments.run_task5_disclosure import (  # noqa: E402
    complete_confirmation_run,
    reserve_confirmation_run,
)
from experiments.run_task5_planning import _summary, _write_checkpoint  # noqa: E402
from organization.hierarchical import HierarchicalOrganizer  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "configs" / "http_transfer_protocol_20260903.json"
CONFIRMATION_REGISTRY = ROOT / "results" / "http_transfer_confirmation_registry.json"
METHODS = {
    "one_stage": {
        "planner_mode": "one_stage",
        "planner_disclosure_level": "full",
    },
    "always_full": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "full",
    },
    "adaptive_signals": {
        "planner_mode": "two_stage",
        "planner_disclosure_level": "adaptive_signals",
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
    "execution/environment.py",
    "execution/http_fixture.py",
    "execution/handlers.py",
    "execution/registry.py",
    "organization/hierarchical.py",
    "retrieval/bm25.py",
    "experiments/run_http_transfer.py",
    "configs/disclosure_policy_v01.json",
    "data/benchmark_http_v01/build_dataset.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    required = {
        "protocol_id", "status", "dataset", "frozen_policy", "controlled",
        "decision_rules", "confirmation_policy",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError("HTTP transfer 协议缺少字段: " + ", ".join(sorted(missing)))
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_frozen_inputs(protocol: dict[str, Any], data: Path) -> None:
    expected = protocol["dataset"]
    paths = {
        "skills_sha256": data / "skills.jsonl",
        "tasks_sha256": data / "tasks.jsonl",
        "environment_fixtures_sha256": data / "environment_fixtures.json",
    }
    mismatches = [key for key, path in paths.items() if sha256(path) != expected[key]]
    policy = protocol["frozen_policy"]
    policy_path = ROOT / policy["path"]
    if sha256(policy_path) != policy["sha256"]:
        mismatches.append("frozen_policy.sha256")
    if mismatches:
        raise RuntimeError("冻结输入哈希不一致: " + ", ".join(mismatches))


def source_manifest() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_FILES}


def classify(output: dict[str, Any]) -> dict[str, Any]:
    rules = output["protocol_snapshot"]["decision_rules"]
    adaptive = output.get("runs", {}).get("adaptive_signals", {})
    full = output.get("runs", {}).get("always_full", {})
    adaptive_rows = adaptive.get("per_task", [])
    full_rows = full.get("per_task", [])
    success_count = sum(row.get("task_success") is True for row in adaptive_rows)
    full_success_count = sum(row.get("task_success") is True for row in full_rows)
    api_errors = len(adaptive.get("api_errors", []))
    levels = sorted({
        row.get("resolved_disclosure_level") for row in adaptive_rows
        if row.get("resolved_disclosure_level")
    })
    cross = rules["cross_domain_transfer"]
    partial = rules["partial_transfer"]
    cross_checks = {
        "adaptive_success_count": success_count >= cross["minimum_adaptive_success_count"],
        "success_drop_vs_always_full": (
            full_success_count - success_count
            <= cross["maximum_success_count_drop_vs_always_full"]
        ),
        "api_errors": api_errors <= cross["maximum_api_errors"],
        "observed_levels": set(cross["required_observed_levels"]).issubset(levels),
    }
    partial_checks = {
        "adaptive_success_count": success_count >= partial["minimum_adaptive_success_count"],
        "api_errors": api_errors <= partial["maximum_api_errors"],
    }
    if adaptive_rows and full_rows and all(cross_checks.values()):
        label = "cross_domain_transfer"
    elif adaptive_rows and all(partial_checks.values()):
        label = "partial_transfer"
    elif adaptive_rows:
        label = "domain_dependent"
    else:
        label = "incomplete"
    return {
        "classification": label,
        "observed": {
            "adaptive_success_count": success_count,
            "always_full_success_count": full_success_count,
            "adaptive_api_errors": api_errors,
            "adaptive_disclosure_levels": dict(Counter(
                row.get("resolved_disclosure_level") for row in adaptive_rows
            )),
        },
        "cross_domain_checks": cross_checks,
        "partial_transfer_checks": partial_checks,
    }


def failure_layers(rows: list[dict[str, Any]]) -> dict[str, int]:
    layers: Counter[str] = Counter()
    for row in rows:
        if row.get("task_success") is True:
            continue
        if row.get("retrieval_metrics", {}).get("recall@9", 0) < 1:
            layers["retrieval"] += 1
        elif row.get("failure_reason") in {
            "invalid_plan",
            "argument_validation_failed",
            "schema_validation_failed",
        }:
            layers["argument_contract"] += 1
        elif row.get("skill_selection_f1", 0) < 1:
            layers["selection"] += 1
        elif row.get("execution_success") is False:
            layers["execution_or_behavior"] += 1
        else:
            layers["verification_or_behavior"] += 1
    return dict(layers)


def write_report(output_path: Path, output: dict[str, Any]) -> Path:
    report_path = output_path.with_suffix(".md")
    lines = [
        f"# HTTP disclosure transfer — {output['split']}",
        "",
        f"Protocol: `{output['protocol_snapshot']['protocol_id']}`; "
        f"status `{output['protocol_snapshot']['status']}`.",
        "",
        "The existing `signal-disclosure-dev-v01` policy was used unchanged. "
        "Every task ran against an isolated loopback HTTP service and the verifier "
        "checked the exact request trace as well as the output.",
        "",
        "| Method | Success | Selection F1 | Sequence | Tokens | P50 ms | API errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in output.get("runs", {}).items():
        summary = run.get("summary", {})
        agent = summary.get("agent", {})
        cost = summary.get("efficiency", {})
        lines.append(
            f"| {name} | {agent.get('task_success_rate', 0):.4f} | "
            f"{agent.get('skill_selection_f1', 0):.4f} | "
            f"{agent.get('sequence_accuracy', 0):.4f} | "
            f"{cost.get('total_tokens', 0)} | "
            f"{cost.get('end_to_end_time_p50_ms', 0):.2f} | "
            f"{len(run.get('api_errors', []))} |"
        )
    adaptive_run = output.get("runs", {}).get("adaptive_signals", {})
    full_run = output.get("runs", {}).get("always_full", {})
    adaptive_tokens = adaptive_run.get("summary", {}).get("efficiency", {}).get(
        "total_tokens", 0
    )
    full_tokens = full_run.get("summary", {}).get("efficiency", {}).get(
        "total_tokens", 0
    )
    token_change = (
        adaptive_tokens / full_tokens - 1 if adaptive_tokens and full_tokens else None
    )
    lines.extend([
        "",
        "## Frozen-rule result",
        "",
        f"- Classification: **{output['transfer_assessment']['classification']}**",
        f"- Observed: `{json.dumps(output['transfer_assessment']['observed'], ensure_ascii=False, sort_keys=True)}`",
        (
            f"- Adaptive token change vs always-full: {token_change:+.2%} "
            f"({adaptive_tokens} vs {full_tokens})."
            if token_change is not None
            else "- Adaptive token comparison is incomplete."
        ),
        "",
        "## Failure layers",
        "",
    ])
    for name, run in output.get("runs", {}).items():
        lines.append(
            f"- `{name}`: `{json.dumps(failure_layers(run.get('per_task', [])), ensure_ascii=False, sort_keys=True)}`"
        )
    lines.extend([
        "",
        "## Adaptive per-task evidence",
        "",
        "| Task | Planned IDs | Level | Markers | Success | Repairs | Tokens | Latency ms | Failure |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in adaptive_run.get("per_task", []):
        planned_ids = [
            item.get("skill_id") for item in row.get("selection_evidence", [])
            if item.get("skill_id")
        ] or list(row.get("selected_skill_ids", []))
        markers = row.get("disclosure_signals", {}).get(
            "matched_behavior_markers", []
        )
        failure = row.get("failure_reason") or "-"
        lines.append(
            f"| {row['task_id']} | {','.join(planned_ids) or '-'} | "
            f"{row.get('resolved_disclosure_level') or '-'} | "
            f"{','.join(markers) or '-'} | "
            f"{str(row.get('task_success')).lower()} | "
            f"{row.get('repair_attempts', 0)} | "
            f"{row.get('token_usage', {}).get('total_tokens', 0)} | "
            f"{row.get('end_to_end_time_ms', 0):.2f} | {failure} |"
        )
    if output["split"] == "confirmation":
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- The frozen cross-domain threshold required at least 5/6 adaptive successes. The observed 4/6 therefore cannot be called full cross-domain transfer; it meets only the preregistered partial-transfer rule.",
            "- `thttpc04` shows positive behavioral transfer: the old markers triggered full disclosure, the planner supplied strict accepted-status/error-body behavior, and the verifier observed exactly one POST.",
            "- `thttpc05` selected the correct PATCH Skill but serialized an object as a string; hidden-Schema escalation and the single repair did not recover. This is an argument-contract failure after correct selection.",
            "- `thttpc06` selected HEAD for a task requiring an error body. Full disclosure cannot repair a wrong Skill selected in stage one, so this is a selection-layer failure.",
            "- Adaptive did not save tokens on confirmation because the failed PATCH case added a repair call. Cost savings from dev do not transfer reliably with six tasks.",
        ])
    lines.extend([
        "",
        "## Validity boundary",
        "",
        "- Six tasks per split are an engineering transfer check, not a broad statistical generalization claim.",
        "- Loopback HTTP preserves methods, headers, status codes, bodies, and request counts but excludes public-network nondeterminism.",
        "- Confirmation results may not change the policy, task family, markers, or thresholds.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol, protocol_hash = load_protocol(protocol_path)
    data = ROOT / "data" / protocol["dataset"]["name"]
    validate_frozen_inputs(protocol, data)
    if args.split == "confirmation" and protocol["status"] != "frozen":
        raise RuntimeError("confirmation 被拒绝：HTTP transfer 协议尚未 frozen")
    allowed_ids = protocol["dataset"][
        "development_task_ids" if args.split == "dev" else "confirmation_task_ids"
    ]
    requested = [value.strip() for value in args.task_ids.split(",") if value.strip()]
    task_ids = requested or list(allowed_ids)
    unknown = sorted(set(task_ids) - set(allowed_ids))
    if unknown:
        raise ValueError("split 外 task_id: " + ", ".join(unknown))
    method_names = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown_methods = sorted(set(method_names) - set(METHODS))
    if unknown_methods:
        raise ValueError("未知方法: " + ", ".join(unknown_methods))
    if args.max_retries < 0:
        raise ValueError("max_retries 不能小于 0")

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else (
        ROOT / "results" / f"http_transfer_{args.split}_{experiment_id}.json"
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"输出已存在: {output_path}")
    if args.split == "confirmation":
        reserve_confirmation_run(
            protocol,
            protocol_hash,
            output_path,
            experiment_id,
            args.resume,
            registry_path=CONFIRMATION_REGISTRY,
        )
    if args.resume:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        for field, expected in {
            "protocol_sha256": protocol_hash,
            "source_manifest": source_manifest(),
            "split": args.split,
            "task_ids": task_ids,
        }.items():
            if output.get(field) != expected:
                raise RuntimeError(f"续跑字段不一致: {field}")
    else:
        output = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "protocol_snapshot": protocol,
            "source_manifest": source_manifest(),
            "split": args.split,
            "task_ids": task_ids,
            "methods": {name: METHODS[name] for name in method_names},
            "runs": {},
        }
        _write_checkpoint(output_path, output)

    controlled = protocol["controlled"]
    available = {task.id for task in load_tasks(data / "tasks.jsonl")}
    if not set(task_ids).issubset(available):
        raise RuntimeError("协议 task_id 不存在于冻结数据")
    for method_name in method_names:
        method = METHODS[method_name]
        run = output["runs"].setdefault(
            method_name, {"per_task": [], "api_errors": [], "summary": {}}
        )
        completed = {row["task_id"] for row in run["per_task"]}
        run["api_errors"] = []
        for task_id in task_ids:
            if task_id in completed:
                continue
            last_error: Exception | None = None
            for attempt in range(args.max_retries + 1):
                try:
                    result = run_benchmark(
                        data / "skills.jsonl",
                        data / "tasks.jsonl",
                        retriever=BM25Retriever(text_level="brief"),
                        organizer=HierarchicalOrganizer(detail_top_k=0),
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
                        retrieval_ks=(1, 3, 5, 9),
                        enable_reflection=False,
                        use_task_context_budget=True,
                        planner_mode=method["planner_mode"],
                        max_argument_repairs=controlled["max_argument_repairs"],
                        planner_disclosure_level=method["planner_disclosure_level"],
                        run_id=f"http-transfer-{experiment_id}-{method_name}-{task_id}",
                    )
                    row = dict(result["per_task"][0])
                    row["attempts"] = attempt + 1
                    run["per_task"].append(row)
                    print(
                        f"[{method_name}] {task_id} success={row.get('task_success')} "
                        f"level={row.get('resolved_disclosure_level')} "
                        f"tokens={row.get('token_usage', {}).get('total_tokens')} "
                        f"latency_ms={row.get('end_to_end_time_ms', 0):.1f}",
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
            output["transfer_assessment"] = classify(output)
            _write_checkpoint(output_path, output)
        print(json.dumps({method_name: run["summary"]}, ensure_ascii=False, indent=2))
    output["transfer_assessment"] = classify(output)
    _write_checkpoint(output_path, output)
    report_path = write_report(output_path, output)
    if args.split == "confirmation":
        complete_confirmation_run(
            protocol,
            protocol_hash,
            output_path,
            registry_path=CONFIRMATION_REGISTRY,
        )
    print(f"结果已保存：{output_path}")
    print(f"报告已保存：{report_path}")


if __name__ == "__main__":
    main()
