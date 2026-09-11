"""Run the frozen CPU dual-encoder engineering diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from data.dataset import RetrievalDataset
from data.loader import load_skills, load_tasks
from retrieval.evaluator import aggregate_ranking_metrics
from retrieval.model import RetrievalModel
from retrieval.train import train

DEFAULT_PROTOCOL = ROOT / "configs" / "dual_encoder_training_protocol_20260906.json"
SOURCE_PATHS = (
    "configs/default.yaml",
    "data/dataset.py",
    "main.py",
    "retrieval/model.py",
    "retrieval/train.py",
    "retrieval/evaluator.py",
    "experiments/run_dual_encoder_training.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: RetrievalModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.net.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def ranking_snapshot(
    model: RetrievalModel,
    tasks: list,
    skills: list,
    top_k: int,
) -> dict[str, Any]:
    model.index(skills)
    rankings: list[tuple[list[str], list[str]]] = []
    per_task = []
    for task in tasks:
        result = model.retrieve(task.instruction, top_k=top_k)
        ranked_ids = result.ranked_ids()
        rankings.append((ranked_ids, list(task.expected_skills)))
        per_task.append(
            {
                "task_id": task.id,
                "ranked_ids": ranked_ids,
                "scores": result.scores,
                "gold_ids": list(task.expected_skills),
            }
        )
    return {
        "metrics": aggregate_ranking_metrics(rankings, ks=(1, 5, 10)),
        "per_task": per_task,
    }


def load_and_validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError("dual-encoder protocol 必须为 frozen")
    if protocol.get("external_api_calls_allowed") is not False:
        raise ValueError("本实验禁止外部 API 调用")
    data = protocol["data"]
    if data.get("allowed_split") != "dev":
        raise ValueError("本工程诊断只允许 benchmark_v01 dev")
    for key in ("skills", "tasks"):
        artifact = ROOT / data[f"{key}_path"]
        actual = sha256_file(artifact)
        if actual != data[f"{key}_sha256"]:
            raise ValueError(f"{key} 数据 SHA-256 与冻结协议不一致")
    return protocol, hashlib.sha256(raw).hexdigest()


def build_model(protocol: dict[str, Any]) -> RetrievalModel:
    config = protocol["model"]
    return RetrievalModel(
        dim=int(config["dim"]),
        vocab_size=int(config["vocab_size"]),
        text_level=str(config["text_level"]),
        seed=int(config["initialization_seed"]),
    )


def train_once(
    protocol: dict[str, Any],
    tasks: list,
    skills: list,
) -> tuple[RetrievalModel, dict[str, Any], dict[str, Any], float]:
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    model = build_model(protocol)
    before = ranking_snapshot(model, tasks, skills, int(evaluation["top_k"]))
    dataset = RetrievalDataset(
        tasks,
        skills,
        num_negatives=int(training["num_explicit_negatives"]),
    )
    started = time.perf_counter()
    train(
        dataset,
        model,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]),
        temperature=float(training["temperature"]),
        seed=int(training["shuffle_seed"]),
        device=str(training["device"]),
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    after = ranking_snapshot(model, tasks, skills, int(evaluation["top_k"]))
    return model, before, after, elapsed


def compare_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rankings_exact = True
    max_score_delta = 0.0
    for left_task, right_task in zip(left["per_task"], right["per_task"]):
        rankings_exact &= left_task["task_id"] == right_task["task_id"]
        rankings_exact &= left_task["ranked_ids"] == right_task["ranked_ids"]
        for left_score, right_score in zip(left_task["scores"], right_task["scores"]):
            max_score_delta = max(max_score_delta, abs(left_score - right_score))
    return {
        "rankings_exact": rankings_exact,
        "max_score_abs_delta": max_score_delta,
        "metrics_exact": left["metrics"] == right["metrics"],
    }


def render_report(result: dict[str, Any]) -> str:
    before = result["repetitions"][0]["before"]["metrics"]
    after = result["repetitions"][0]["after"]["metrics"]
    losses = result["repetitions"][0]["loss_history"]
    checks = result["acceptance_checks"]
    rows = [
        "# Dual-encoder dev engineering diagnostic",
        "",
        f"- Protocol: `{result['protocol_id']}`",
        f"- Protocol SHA-256: `{result['protocol_sha256']}`",
        f"- Data: benchmark_v01 dev only ({result['task_count']} tasks, {result['sample_count']} positive samples)",
        "- Claim boundary: training-set engineering diagnostic; no test/confirmation task was selected or evaluated.",
        "",
        "## Result",
        "",
        f"- Loss: {losses[0]:.6f} -> {losses[-1]:.6f}",
        f"- MRR: {before['mrr']:.6f} -> {after['mrr']:.6f}",
        f"- Recall@1: {before['recall@1']:.6f} -> {after['recall@1']:.6f}",
        f"- Recall@5: {before['recall@5']:.6f} -> {after['recall@5']:.6f}",
        f"- Checkpoint SHA-256: `{result['checkpoint']['sha256']}`",
        f"- Checkpoint reload max score delta: {result['checkpoint']['roundtrip']['max_score_abs_delta']:.12f}",
        f"- Two-run state hash exact: {checks['repeated_state_hash_exact']}",
        f"- Two-run loss history exact: {checks['repeated_loss_history_exact']}",
        f"- Overall acceptance: **{result['accepted']}**",
        "",
        "## Interpretation boundary",
        "",
        "This run verifies optimization, deterministic repetition, stale-index invalidation, and portable weight reload. Because the same dev tasks supply training and evaluation, the metric gain is not evidence of held-out quality or generalization.",
        "",
    ]
    return "\n".join(rows)


def run(protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_and_validate_protocol(protocol_path)
    torch.set_num_threads(1)
    data = protocol["data"]
    skills = load_skills(ROOT / data["skills_path"])
    all_tasks = load_tasks(ROOT / data["tasks_path"])
    tasks = [
        task for task in all_tasks
        if task.metadata.get("split") == data["allowed_split"]
    ]
    if not tasks or any(task.metadata.get("split") != "dev" for task in tasks):
        raise ValueError("任务选择越过了冻结 dev 边界")
    sample_count = len(
        RetrievalDataset(
            tasks,
            skills,
            num_negatives=int(protocol["training"]["num_explicit_negatives"]),
        )
    )

    repetitions = []
    models = []
    for repetition in range(int(protocol["training"]["independent_repetitions"])):
        model, before, after, elapsed = train_once(protocol, tasks, skills)
        models.append(model)
        repetitions.append(
            {
                "repetition": repetition + 1,
                "before": before,
                "after": after,
                "loss_history": model.training_history,
                "state_sha256": state_hash(model),
                "training_seconds": elapsed,
            }
        )

    output = protocol["outputs"]
    checkpoint_path = ROOT / output["checkpoint"]
    models[0].save_checkpoint(
        checkpoint_path,
        metadata={
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "scope": "benchmark_v01-dev-resubstitution",
        },
    )
    restored = RetrievalModel.load_checkpoint(checkpoint_path, map_location="cpu")
    restored_snapshot = ranking_snapshot(
        restored,
        tasks,
        skills,
        int(protocol["evaluation"]["top_k"]),
    )
    roundtrip = compare_snapshots(repetitions[0]["after"], restored_snapshot)

    acceptance = protocol["evaluation"]["acceptance"]
    checks = {
        "final_loss_below_initial_loss": (
            repetitions[0]["loss_history"][-1]
            < repetitions[0]["loss_history"][0]
        ),
        "post_train_mrr_above_pre_train_mrr": (
            repetitions[0]["after"]["metrics"]["mrr"]
            > repetitions[0]["before"]["metrics"]["mrr"]
        ),
        "post_train_recall_at_5_min": (
            repetitions[0]["after"]["metrics"]["recall@5"]
            >= float(acceptance["post_train_recall_at_5_min"])
        ),
        "repeated_loss_history_exact": (
            repetitions[0]["loss_history"] == repetitions[1]["loss_history"]
        ),
        "repeated_state_hash_exact": (
            repetitions[0]["state_sha256"] == repetitions[1]["state_sha256"]
        ),
        "checkpoint_rankings_exact": roundtrip["rankings_exact"],
        "checkpoint_scores_max_abs_delta": (
            roundtrip["max_score_abs_delta"]
            <= float(acceptance["checkpoint_scores_max_abs_delta"])
        ),
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "claim_boundary": protocol["claim_boundary"],
        "data_manifest": {
            "skills_path": data["skills_path"],
            "skills_sha256": data["skills_sha256"],
            "tasks_path": data["tasks_path"],
            "tasks_sha256": data["tasks_sha256"],
            "loaded_split": data["allowed_split"],
            "prohibited_splits_selected": [],
        },
        "source_manifest": {
            path: sha256_file(ROOT / path) for path in SOURCE_PATHS
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "task_count": len(tasks),
        "skill_count": len(skills),
        "sample_count": sample_count,
        "model": protocol["model"],
        "training": protocol["training"],
        "repetitions": repetitions,
        "checkpoint": {
            "path": output["checkpoint"],
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "metadata": restored.checkpoint_metadata,
            "roundtrip": roundtrip,
        },
        "acceptance_checks": checks,
        "accepted": all(checks.values()),
    }
    result_path = ROOT / output["result_json"]
    report_path = ROOT / output["report_markdown"]
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = run(args.protocol.resolve())
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "accepted": result["accepted"],
        "result": result["checkpoint"]["path"],
        "checks": result["acceptance_checks"],
    }, ensure_ascii=False, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
