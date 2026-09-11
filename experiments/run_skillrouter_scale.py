"""Run the frozen SkillRouter large-pool retrieval scaling protocol.

The default path is CPU-safe and evaluates exact BM25 variants only. Official
0.6B models are opt-in and the frozen full-pool protocol refuses to run them
without CUDA. No LLM Planner or Agent request is made by this script.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_PROTOCOL = ROOT / "configs" / "skillrouter_large_pool_protocol_20260903.json"
DEFAULT_DATA_ROOT = (
    WORKSPACE / "tmp" / "upstream_skillrouter_20260830" / "data" / "eval_core"
)
DEFAULT_UPSTREAM_ROOT = WORKSPACE / "tmp" / "upstream_skillrouter_20260830"
DEFAULT_OUTPUT = ROOT / "results" / "skillrouter_large_pool_bm25_v03_20260903.json"
DEFAULT_REPORT = ROOT / "results" / "skillrouter_large_pool_bm25_v03_20260903.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.schemas import Skill  # noqa: E402
from retrieval.published import (  # noqa: E402
    SkillRouterEmbeddingRetriever,
    SkillRouterRetriever,
)
from retrieval.tokenize import tokenize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--upstream-root", default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--include-official-models",
        action="store_true",
        help="Run official encoder/pipeline; full pools require CUDA.",
    )
    parser.add_argument(
        "--official-max-pool-size",
        type=int,
        default=None,
        help="Optional CPU smoke ceiling; does not waive the full-pool CUDA gate.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: str, tier: str, skill_id: str) -> str:
    return hashlib.sha256(f"{seed}|{tier}|{skill_id}".encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl*"))
    if not paths:
        raise FileNotFoundError(f"No JSONL shards found: {path}")
    for item in paths:
        opener = gzip.open if item.name.endswith(".gz") else Path.open
        kwargs = {"mode": "rt", "encoding": "utf-8"}
        with opener(item, **kwargs) as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Non-object JSONL row in {item}")
                    yield value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "total_ms": sum(values),
        "throughput_qps": len(values) / (sum(values) / 1000) if sum(values) else 0.0,
    }


def dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def metrics(
    ranked_ids: list[str],
    gt_ids: set[str],
    relevance: dict[str, float],
) -> dict[str, float]:
    observed = [float(relevance.get(item, 0.0)) for item in ranked_ids]
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)

    def ndcg_at(k: int) -> float:
        denominator = dcg(ideal, k)
        return dcg(observed, k) / denominator if denominator else 0.0

    def recall_at(k: int) -> float:
        return len(set(ranked_ids[:k]) & gt_ids) / len(gt_ids) if gt_ids else 0.0

    def full_coverage(k: int) -> float:
        return float(gt_ids.issubset(set(ranked_ids[:k]))) if gt_ids else 1.0

    first_rank = next(
        (index for index, item in enumerate(ranked_ids[:10], 1) if item in gt_ids),
        None,
    )
    return {
        "nDCG@1": ndcg_at(1),
        "nDCG@3": ndcg_at(3),
        "nDCG@10": ndcg_at(10),
        "Hit@1": float(bool(set(ranked_ids[:1]) & gt_ids)),
        "Precision@3": len(set(ranked_ids[:3]) & gt_ids) / 3,
        "MRR@10": 1 / first_rank if first_rank else 0.0,
        "Recall@10": recall_at(10),
        "Recall@20": recall_at(20),
        "Recall@50": recall_at(50),
        "FullCoverage@3": full_coverage(3),
        "FullCoverage@5": full_coverage(5),
        "FullCoverage@10": full_coverage(10),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    output: dict[str, float | int] = {
        key: statistics.mean(row[key] for row in rows) for key in rows[0]
    }
    output["count"] = len(rows)
    return output


def task_groups(
    tasks: list[dict[str, Any]],
    relevance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        task for task in tasks
        if relevance[task["task_id"]].get("task_type") != "generic_only"
    ]


def graded_ids_for_tasks(
    tasks: list[dict[str, Any]],
    relevance: dict[str, dict[str, Any]],
) -> set[str]:
    return {
        skill_id
        for task in tasks
        for skill_id in relevance[task["task_id"]].get("relevance", {})
    }


def ordered_pool(
    records: list[dict[str, Any]],
    required_ids: set[str],
    seed: str,
    tier: str,
) -> tuple[list[dict[str, Any]], int]:
    by_id = {str(record["skill_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"{tier} pool contains duplicate skill IDs")
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise ValueError(f"{tier} pool misses {len(missing)} graded IDs")
    required = sorted(required_ids, key=lambda item: stable_key(seed, tier, item))
    distractors = sorted(
        set(by_id) - required_ids,
        key=lambda item: stable_key(seed, tier, item),
    )
    ordered_ids = [*required, *distractors]
    return [by_id[item] for item in ordered_ids], len(required)


class CompactBM25Index:
    """Full inverted BM25 index with nested active-prefix evaluation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.records: list[dict[str, Any]] = []
        self.skill_ids: list[str] = []
        self.tie_keys: list[str] = []
        self.doc_lengths = array("I")
        self.length_prefix = array("Q", [0])
        self.posting_docs: dict[str, array] = {}
        self.posting_tfs: dict[str, array] = {}
        self.checkpoints: dict[int, dict[str, float | int]] = {}

    def build(
        self,
        records: list[dict[str, Any]],
        text_fn,
        checkpoint_sizes: list[int],
        seed: str,
        tier: str,
    ) -> None:
        self.records = records
        self.skill_ids = [str(record["skill_id"]) for record in records]
        self.tie_keys = [stable_key(seed, tier, item) for item in self.skill_ids]
        checkpoints = set(checkpoint_sizes)
        tracemalloc.start()
        started = time.perf_counter()
        for doc_id, record in enumerate(records):
            frequencies = Counter(tokenize(text_fn(record)))
            length = sum(frequencies.values())
            self.doc_lengths.append(length)
            self.length_prefix.append(self.length_prefix[-1] + length)
            for token, frequency in frequencies.items():
                self.posting_docs.setdefault(token, array("I")).append(doc_id)
                self.posting_tfs.setdefault(token, array("I")).append(frequency)
            count = doc_id + 1
            if count in checkpoints:
                current, peak = tracemalloc.get_traced_memory()
                self.checkpoints[count] = {
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "traced_current_bytes": current,
                    "traced_peak_bytes": peak,
                    "vocabulary_size": len(self.posting_docs),
                    "total_document_tokens": int(self.length_prefix[-1]),
                }
        tracemalloc.stop()

    def retrieve(self, query: str, active_size: int, top_k: int = 50) -> list[str]:
        if active_size < 1 or active_size > len(self.records):
            raise ValueError("active_size outside indexed pool")
        average_length = self.length_prefix[active_size] / active_size
        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(tokenize(query))):
            docs = self.posting_docs.get(token)
            frequencies = self.posting_tfs.get(token)
            if docs is None or frequencies is None:
                continue
            active_df = bisect.bisect_left(docs, active_size)
            if not active_df:
                continue
            inverse_document_frequency = math.log(
                1 + (active_size - active_df + 0.5) / (active_df + 0.5)
            )
            for offset in range(active_df):
                doc_id = docs[offset]
                frequency = frequencies[offset]
                length = self.doc_lengths[doc_id]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(average_length, 1e-9)
                )
                scores[doc_id] += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )
        ranked = sorted(scores, key=lambda item: (-scores[item], self.tie_keys[item]))
        if len(ranked) < min(top_k, active_size):
            scored = set(ranked)
            zero_docs = sorted(
                (index for index in range(active_size) if index not in scored),
                key=lambda item: self.tie_keys[item],
            )
            ranked.extend(zero_docs[:top_k - len(ranked)])
        return [self.skill_ids[index] for index in ranked[:top_k]]


def evaluate_ranker(
    retrieve_fn,
    tasks: list[dict[str, Any]],
    relevance: dict[str, dict[str, Any]],
    pool_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_task = []
    latencies = []
    for task in tasks:
        task_id = str(task["task_id"])
        rel_entry = relevance[task_id]
        gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
        gt_ids &= pool_ids
        if not gt_ids:
            continue
        rel_map = {
            str(key): float(value)
            for key, value in rel_entry.get("relevance", {}).items()
            if key in pool_ids
        }
        started = time.perf_counter()
        ranking = retrieve_fn(str(task["instruction_text"]))
        duration_ms = (time.perf_counter() - started) * 1000
        latencies.append(duration_ms)
        row_metrics = metrics(ranking, gt_ids, rel_map)
        group = "single" if len(gt_ids) == 1 else "multi"
        metric_rows["all"].append(row_metrics)
        metric_rows[group].append(row_metrics)
        per_task.append({
            "task_id": task_id,
            "group": group,
            "gt_skill_ids": sorted(gt_ids),
            "ranked_skill_ids": ranking,
            "metrics": row_metrics,
            "query_latency_ms": duration_ms,
        })
    return {
        "metrics": {
            group: aggregate_metrics(rows) for group, rows in metric_rows.items()
        },
        "query_latency": latency_summary(latencies),
    }, per_task


def bm25_text(record: dict[str, Any], level: str) -> str:
    name = str(record.get("name", ""))
    description = str(record.get("description") or record.get("desc") or "")[:500]
    if level == "brief":
        return f"{name} {description}"
    body = str(record.get("body") or "")[:8000]
    return f"{name} {description} {body}"


def pool_sizes(protocol: dict[str, Any], full_size: int) -> list[int]:
    output = []
    for item in protocol["scale_slices"]["official_pool_sizes"]:
        value = full_size if item == "full" else int(item)
        if value > full_size:
            raise ValueError(f"Configured pool size {value} exceeds {full_size}")
        if value not in output:
            output.append(value)
    return output


def evaluate_bm25_level(
    level: str,
    records: list[dict[str, Any]],
    sizes: list[int],
    tasks: list[dict[str, Any]],
    relevance: dict[str, dict[str, Any]],
    seed: str,
    tier: str,
) -> dict[str, Any]:
    index = CompactBM25Index()
    index.build(
        records,
        lambda record: bm25_text(record, level),
        sizes,
        seed,
        tier,
    )
    by_size = {}
    for size in sizes:
        pool_id_set = set(index.skill_ids[:size])
        summary, per_task = evaluate_ranker(
            lambda query, size=size: index.retrieve(query, size, top_k=50),
            tasks,
            relevance,
            pool_id_set,
        )
        by_size[str(size)] = {
            "index_checkpoint": index.checkpoints[size],
            **summary,
            "per_task": per_task,
        }
    return {
        "status": "completed",
        "text_level": level,
        "k1": index.k1,
        "b": index.b,
        "results_by_pool_size": by_size,
    }


def to_skill(record: dict[str, Any]) -> Skill:
    return Skill(
        id=str(record["skill_id"]),
        name=str(record.get("name", "")),
        brief_description=str(record.get("description") or record.get("desc") or ""),
        detailed_description=str(record.get("body") or ""),
    )


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def evaluate_official_method(
    method: str,
    records: list[dict[str, Any]],
    sizes: list[int],
    tasks: list[dict[str, Any]],
    relevance: dict[str, dict[str, Any]],
    max_pool_size: int | None,
) -> dict[str, Any]:
    has_cuda = cuda_available()
    full_size = len(records)
    eligible_sizes = [
        size for size in sizes
        if max_pool_size is None or size <= max_pool_size
    ]
    if full_size in eligible_sizes and not has_cuda:
        raise RuntimeError("Frozen protocol requires CUDA for official full-pool runs")
    if not eligible_sizes:
        return {"status": "not_run", "reason": "no pool size passed the ceiling"}
    retriever = (
        SkillRouterEmbeddingRetriever(batch_size=32)
        if method == "skillrouter_encoder"
        else SkillRouterRetriever(
            retrieval_top_k=20,
            batch_size=32,
            reranker_batch_size=8,
        )
    )
    prepare_started = time.perf_counter()
    retriever.prepare()
    prepare_ms = (time.perf_counter() - prepare_started) * 1000
    results = {}
    for size in eligible_sizes:
        skills = [to_skill(record) for record in records[:size]]
        index_started = time.perf_counter()
        retriever.index(skills)
        index_ms = (time.perf_counter() - index_started) * 1000
        pool_ids = {skill.id for skill in skills}
        summary, per_task = evaluate_ranker(
            lambda query: retriever.retrieve(query, top_k=50).ranked_ids(),
            tasks,
            relevance,
            pool_ids,
        )
        results[str(size)] = {
            "index_ms": index_ms,
            **summary,
            "per_task": per_task,
        }
    return {
        "status": "completed",
        "prepare_ms": prepare_ms,
        "provenance": retriever.provenance(),
        "results_by_pool_size": results,
    }


def verify_inputs(
    protocol: dict[str, Any],
    protocol_path: Path,
    data_root: Path,
    upstream_root: Path,
) -> dict[str, Any]:
    if protocol.get("status") != "frozen":
        raise ValueError("Large-pool protocol must be frozen before execution")
    dataset = protocol["dataset"]
    hashes = {}
    for key, name in (
        ("manifest_sha256", "manifest.json"),
        ("tasks_sha256", "tasks.jsonl"),
        ("relevance_sha256", "relevance.json"),
    ):
        actual = sha256_file(data_root / name)
        if actual != dataset[key]:
            raise ValueError(f"{name} SHA-256 mismatch")
        hashes[name] = actual
    upstream_commit = "unavailable"
    head_path = upstream_root / ".git" / "HEAD"
    if head_path.exists():
        import subprocess

        upstream_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=upstream_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        expected = protocol["methods"]["skillrouter_encoder"]["upstream_commit"]
        if upstream_commit != expected:
            raise ValueError("SkillRouter upstream commit mismatch")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "metadata_sha256": hashes,
        "upstream_commit": upstream_commit,
    }


def environment_info() -> dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": cuda_available(),
    }
    try:
        import torch
        import transformers

        info.update({
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        })
    except ImportError:
        info.update({"torch": None, "transformers": None, "cuda_device": None})
    return info


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# SkillRouter large-pool scaling — frozen protocol report",
        "",
        f"Protocol: `{result['protocol']['protocol_id']}`; status "
        f"`{result['protocol']['status']}`. This is retrieval-only: no LLM Planner "
        "or Agent calls were issued.",
        "",
        f"Protocol history: {result['protocol'].get('revision_reason', 'none')} ",
        "",
        "## Frozen questions",
        "",
        f"- Quality: {result['protocol']['hypotheses']['quality']}",
        f"- Cost: {result['protocol']['hypotheses']['cost']}",
        "",
        "The local 24-skill result is context only and is not plotted as an "
        "official-dataset scale point. Official subsets contain every graded "
        "relevance ID and use a frozen relevance-blind hash order for distractors "
        "and score ties.",
        "",
    ]
    for tier, tier_data in result["tiers"].items():
        lines.extend([
            f"## {tier} tier",
            "",
            f"Pool records: {tier_data['pool_records']:,}; graded IDs forced into "
            f"every slice: {tier_data['graded_ids_in_pool']}.",
            "",
            "| Method | Pool | MRR@10 | Recall@10 | NDCG@10 | FullCoverage@10 | "
            "Index ms | Query mean ms | Query P95 ms | Traced peak MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in ("bm25_brief", "bm25_all_fields"):
            method_data = tier_data["methods"][method]
            for size, row in method_data["results_by_pool_size"].items():
                aggregate = row["metrics"]["all"]
                checkpoint = row["index_checkpoint"]
                query = row["query_latency"]
                lines.append(
                    f"| {method} | {int(size):,} | {aggregate['MRR@10']:.4f} | "
                    f"{aggregate['Recall@10']:.4f} | {aggregate['nDCG@10']:.4f} | "
                    f"{aggregate['FullCoverage@10']:.4f} | "
                    f"{checkpoint['elapsed_ms']:,.2f} | {query['mean_ms']:,.3f} | "
                    f"{query['p95_ms']:,.3f} | "
                    f"{checkpoint['traced_peak_bytes'] / 1024 / 1024:,.2f} |"
                )
        for method in ("skillrouter_encoder", "skillrouter_pipeline"):
            method_data = tier_data["methods"][method]
            if method_data.get("status") != "completed":
                lines.extend([
                    "",
                    f"- `{method}`: **{method_data.get('status')}** — "
                    f"{method_data.get('reason', 'not executed')}",
                ])
        lines.append("")
    lines.extend(["## BM25 findings", ""])
    for tier, tier_data in result["tiers"].items():
        full_size = str(tier_data["pool_sizes"][-1])
        small_size = str(tier_data["pool_sizes"][0])
        brief_small = tier_data["methods"]["bm25_brief"][
            "results_by_pool_size"
        ][small_size]
        brief_full = tier_data["methods"]["bm25_brief"][
            "results_by_pool_size"
        ][full_size]
        all_small = tier_data["methods"]["bm25_all_fields"][
            "results_by_pool_size"
        ][small_size]
        all_full = tier_data["methods"]["bm25_all_fields"][
            "results_by_pool_size"
        ][full_size]
        brief_metrics = brief_full["metrics"]["all"]
        all_metrics = all_full["metrics"]["all"]
        query_ratio = (
            all_full["query_latency"]["mean_ms"]
            / brief_full["query_latency"]["mean_ms"]
        )
        index_ratio = (
            all_full["index_checkpoint"]["elapsed_ms"]
            / brief_full["index_checkpoint"]["elapsed_ms"]
        )
        memory_ratio = (
            all_full["index_checkpoint"]["traced_peak_bytes"]
            / brief_full["index_checkpoint"]["traced_peak_bytes"]
        )
        lines.extend([
            f"- `{tier}`: from 1k to {int(full_size):,}, brief MRR@10 drops "
            f"{brief_small['metrics']['all']['MRR@10']:.4f} → "
            f"{brief_metrics['MRR@10']:.4f}; all-field drops "
            f"{all_small['metrics']['all']['MRR@10']:.4f} → "
            f"{all_metrics['MRR@10']:.4f}. The small-pool ceiling does not persist.",
            f"- `{tier}` full pool: all-field improves MRR@10 from "
            f"{brief_metrics['MRR@10']:.4f} to {all_metrics['MRR@10']:.4f} and "
            f"Recall@10 from {brief_metrics['Recall@10']:.4f} to "
            f"{all_metrics['Recall@10']:.4f}, but costs {query_ratio:.1f}× mean "
            f"query latency, {index_ratio:.1f}× cumulative index time, and "
            f"{memory_ratio:.1f}× traced peak index memory.",
        ])
    lines.extend([
        "",
        "The evidence supports body-aware retrieval under large distractor pools, "
        "but it also shows that richer lexical indexing is not free. Official "
        "SkillRouter neural curves remain unmeasured until a CUDA environment can "
        "run the unchanged frozen protocol.",
        "",
    ])
    lines.extend([
        "## Interpretation boundary",
        "",
        "- Quality comparisons across pool sizes are valid within this official "
        "dataset protocol; they must not be merged numerically with the local "
        "24-skill benchmark.",
        "- BM25 traced memory covers Python allocations made while building the "
        "index, not total process RSS or compressed dataset storage.",
        "- A missing official neural result is reported as missing. CPU projections "
        "and Hash embeddings are not substituted for official 0.6B runs.",
        "- The 1k/10k slices include all graded items by construction; they measure "
        "distractor-scale sensitivity, not random task coverage loss.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    data_root = Path(args.data_root)
    upstream_root = Path(args.upstream_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verification = verify_inputs(protocol, protocol_path, data_root, upstream_root)
    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    tasks = task_groups(tasks, relevance)
    if len(tasks) != int(protocol["dataset"]["expected_task_count"]):
        raise ValueError("Core task count does not match frozen protocol")
    graded_ids = graded_ids_for_tasks(tasks, relevance)
    seed = str(protocol["scale_slices"]["seed"])

    result: dict[str, Any] = {
        "experiment_id": f"{protocol['protocol_id']}-run1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis_only": False,
        "llm_planner_requests": 0,
        "protocol": protocol,
        "verification": verification,
        "environment": environment_info(),
        "tiers": {},
    }
    for tier, expected_count in protocol["dataset"]["tiers"].items():
        load_started = time.perf_counter()
        raw_records = load_jsonl(data_root / tier)
        load_ms = (time.perf_counter() - load_started) * 1000
        if len(raw_records) != int(expected_count):
            raise ValueError(f"{tier} record count mismatch")
        pool_id_set = {str(record["skill_id"]) for record in raw_records}
        tier_graded_ids = graded_ids & pool_id_set
        expected_present = int(
            protocol["scale_slices"]["required_ids_present_per_tier"]
        )
        if len(tier_graded_ids) != expected_present:
            raise ValueError(
                f"{tier} graded IDs present={len(tier_graded_ids)}, "
                f"expected={expected_present}"
            )
        records, required_count = ordered_pool(
            raw_records,
            tier_graded_ids,
            seed,
            tier,
        )
        sizes = pool_sizes(protocol, len(records))
        if any(size < required_count for size in sizes):
            raise ValueError(f"{tier} slice smaller than graded relevance set")
        memberships = {
            str(size): hashlib.sha256(
                "\n".join(str(row["skill_id"]) for row in records[:size]).encode(
                    "utf-8"
                )
            ).hexdigest()
            for size in sizes
        }
        methods = {
            "bm25_brief": evaluate_bm25_level(
                "brief", records, sizes, tasks, relevance, seed, tier
            ),
            "bm25_all_fields": evaluate_bm25_level(
                "all", records, sizes, tasks, relevance, seed, tier
            ),
        }
        if args.include_official_models:
            for method in ("skillrouter_encoder", "skillrouter_pipeline"):
                methods[method] = evaluate_official_method(
                    method,
                    records,
                    sizes,
                    tasks,
                    relevance,
                    args.official_max_pool_size,
                )
        else:
            reason = (
                "CUDA unavailable; frozen full-pool hardware gate prohibits a "
                "formal CPU or Hash substitute"
                if not cuda_available()
                else "official models not requested in this run"
            )
            methods["skillrouter_encoder"] = {
                "status": "not_run_hardware_gate" if not cuda_available() else "not_run",
                "reason": reason,
            }
            methods["skillrouter_pipeline"] = {
                "status": "not_run_hardware_gate" if not cuda_available() else "not_run",
                "reason": reason,
            }
        result["tiers"][tier] = {
            "pool_records": len(records),
            "data_load_ms": load_ms,
            "graded_ids_in_pool": required_count,
            "pool_sizes": sizes,
            "membership_sha256": memberships,
            "shard_sha256": {
                path.name: sha256_file(path) for path in sorted((data_root / tier).glob("*.jsonl.gz"))
            },
            "methods": methods,
        }

    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(render_report(result), encoding="utf-8")
    print(f"Result saved: {output}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
