"""Run a clearly labeled official SkillRouter encoder CPU smoke on easy-1k."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_skillrouter_scale import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROTOCOL,
    DEFAULT_UPSTREAM_ROOT,
    environment_info,
    evaluate_official_method,
    graded_ids_for_tasks,
    load_jsonl,
    ordered_pool,
    sha256_file,
    task_groups,
    verify_inputs,
)


DEFAULT_OUTPUT = ROOT / "results" / "skillrouter_encoder_easy1k_cpu_smoke_20260903.json"
DEFAULT_REPORT = ROOT / "results" / "skillrouter_encoder_easy1k_cpu_smoke_20260903.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--upstream-root", default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--acknowledge-heavy-cpu-smoke",
        action="store_true",
        help=(
            "Required because the first frozen easy-1k CPU attempt used more than "
            "55 GB of private memory and did not finish in 17 minutes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_heavy_cpu_smoke:
        raise SystemExit(
            "Refusing an accidental heavy CPU rerun. Read "
            "results/skillrouter_encoder_easy1k_cpu_resource_audit_20260903.md "
            "and pass --acknowledge-heavy-cpu-smoke only for an intentional rerun."
        )
    protocol_path = Path(args.protocol)
    data_root = Path(args.data_root)
    upstream_root = Path(args.upstream_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verification = verify_inputs(protocol, protocol_path, data_root, upstream_root)
    if 1000 not in protocol["scale_slices"]["official_pool_sizes"]:
        raise ValueError("Frozen protocol does not contain the easy-1k slice")

    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    tasks = task_groups(tasks, relevance)
    raw_records = load_jsonl(data_root / "easy")
    graded_ids = graded_ids_for_tasks(tasks, relevance)
    pool_ids = {str(record["skill_id"]) for record in raw_records}
    present_graded = graded_ids & pool_ids
    expected_present = int(protocol["scale_slices"]["required_ids_present_per_tier"])
    if len(present_graded) != expected_present:
        raise ValueError("easy graded-ID count differs from frozen protocol")
    records, _ = ordered_pool(
        raw_records,
        present_graded,
        str(protocol["scale_slices"]["seed"]),
        "easy",
    )

    method = evaluate_official_method(
        "skillrouter_encoder",
        records,
        [1000],
        tasks,
        relevance,
        max_pool_size=1000,
    )
    result = {
        "experiment_id": "skillrouter-encoder-easy1k-cpu-smoke-20260903-v01",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "completed_nonformal_cpu_smoke",
        "formal_full_pool_result": False,
        "scope": "Official encoder interface/quality smoke on the frozen easy-1k membership only; not a replacement for CUDA full-pool evaluation.",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "verification": verification,
        "environment": environment_info(),
        "tier": "easy",
        "pool_size": 1000,
        "method": method,
    }
    output = Path(args.output)
    report = Path(args.report)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = method["results_by_pool_size"]["1000"]
    overall = summary["metrics"]["all"]
    latency = summary["query_latency"]
    report.write_text(
        "\n".join([
            "# SkillRouter official encoder easy-1k CPU smoke",
            "",
            "This is a non-formal CPU smoke on the frozen easy-1k membership. It "
            "verifies the official 0.6B encoder path on official data, but it is "
            "not a substitute for the hardware-gated CUDA full-pool run.",
            "",
            f"- Protocol: `{protocol['protocol_id']}`",
            f"- Model: `{method['provenance']['encoder_model']}`",
            f"- Prepare: {method['prepare_ms']:,.2f} ms",
            f"- Index 1,000: {summary['index_ms']:,.2f} ms",
            f"- MRR@10: {overall['MRR@10']:.4f}",
            f"- Recall@10: {overall['Recall@10']:.4f}",
            f"- NDCG@10: {overall['nDCG@10']:.4f}",
            f"- FullCoverage@10: {overall['FullCoverage@10']:.4f}",
            f"- Mean query latency: {latency['mean_ms']:,.2f} ms",
            f"- P95 query latency: {latency['p95_ms']:,.2f} ms",
            "",
            "The official reranker is intentionally not run on CPU: the frozen "
            "local 24-skill evidence already measured roughly 24 seconds per query, "
            "so an easy-1k reranker smoke would add substantial cost without "
            "answering the full-pool scaling question.",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Result saved: {output}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
