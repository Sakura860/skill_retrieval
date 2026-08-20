"""检索器与组织策略消融实验。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PROJECT_ROOT, load_config
from core.llm import LLM
from evaluation.run_benchmark import run_benchmark
from organization.flat import FlatOrganizer
from organization.hierarchical import HierarchicalOrganizer
from organization.graph import GraphOrganizer
from retrieval.bm25 import BM25Retriever
from retrieval.embedding import EmbeddingRetriever
from retrieval.multilevel import MultiLevelRetriever

SKILLS = PROJECT_ROOT / "data" / "skills"
TASKS = PROJECT_ROOT / "data" / "tasks"


def main() -> None:
    retrievers = {
        "bm25": BM25Retriever(),
        "embedding": EmbeddingRetriever(),
        "multilevel": MultiLevelRetriever(),
    }
    organizers = {
        "flat": FlatOrganizer(),
        "hierarchical": HierarchicalOrganizer(),
        "graph": GraphOrganizer(),
    }

    print("=== 统一评测（retriever × organizer）===")
    config = load_config()
    llm_config = config["llm"]
    llm = LLM(
        provider=llm_config["provider"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        thinking=llm_config.get("thinking"),
        reasoning_effort=llm_config.get("reasoning_effort"),
    )
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    runs = []
    for rname, ret in retrievers.items():
        for oname, org in organizers.items():
            run_id = f"{experiment_id}_{rname}_{oname}"
            out = run_benchmark(
                SKILLS,
                TASKS,
                retriever=ret,
                organizer=org,
                llm=llm,
                top_k=config["retrieval"]["top_k"],
                retrieval_ks=tuple(config["evaluation"]["retrieval_ks"]),
                run_id=run_id,
                max_steps=config["agent"]["max_steps"],
                enable_reflection=config["agent"]["enable_reflection"],
            )
            runs.append(out)
            retrieval = out["metrics"]["retrieval"]
            agent = out["metrics"]["agent"]
            efficiency = out["metrics"]["efficiency"]
            success = agent["task_success_rate"]
            success_text = "unscored" if success is None else f"{success:.3f}"
            print(
                f"[{rname} × {oname}] "
                f"recall@5={retrieval.get('recall@5', 0.0):.3f} "
                f"mrr={retrieval.get('mrr', 0.0):.3f} "
                f"selection_f1={agent['skill_selection_f1']:.3f} "
                f"sequence_acc={agent['sequence_accuracy']:.3f} "
                f"success={success_text} "
                f"calls={efficiency['avg_skill_calls']:.2f} "
                f"context_tokens={efficiency['avg_skill_context_tokens']:.1f} "
                f"total_tokens={efficiency['total_tokens']}"
            )

    output = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"experiment_{experiment_id}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完整结果已保存：{output_path}")


if __name__ == "__main__":
    main()
