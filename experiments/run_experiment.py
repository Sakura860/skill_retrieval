"""检索器与组织策略消融实验。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PROJECT_ROOT, load_config
from core.llm import LLM
from data.loader import load_skills, load_tasks
from evaluation.run_benchmark import run_benchmark
from organization.flat import FlatOrganizer
from organization.hierarchical import HierarchicalOrganizer
from organization.graph import GraphOrganizer
from retrieval.bm25 import BM25Retriever
from retrieval.embedding import EmbeddingRetriever
from retrieval.evaluator import evaluate_retrieval
from retrieval.multilevel import MultiLevelRetriever

SKILLS = PROJECT_ROOT / "data" / "skills"
TASKS = PROJECT_ROOT / "data" / "tasks"


def main() -> None:
    skills = load_skills(SKILLS)
    tasks = load_tasks(TASKS)

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

    print("=== 检索召回评测 ===")
    for name, ret in retrievers.items():
        scores = evaluate_retrieval(ret, tasks, skills)
        backend = getattr(ret, "backend", "native")
        print(f"[{name} / {backend}] {scores}")

    print("\n=== 端到端消融（retriever × organizer）===")
    llm_config = load_config()["llm"]
    llm = LLM(
        provider=llm_config["provider"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        thinking=llm_config.get("thinking"),
        reasoning_effort=llm_config.get("reasoning_effort"),
    )
    for rname, ret in retrievers.items():
        for oname, org in organizers.items():
            out = run_benchmark(SKILLS, TASKS, retriever=ret, organizer=org, llm=llm)
            m = out["metrics"]
            print(
                f"[{rname} × {oname}] "
                f"selection_f1={m.skill_selection_f1:.3f} "
                f"sequence_acc={m.sequence_accuracy:.3f} "
                f"success={m.task_success_rate:.3f} "
                f"calls={m.avg_skill_calls:.2f} "
                f"redundant={m.redundant_call_rate:.3f} "
                f"context_tokens={m.skill_context_tokens:.1f} "
                f"total_tokens={m.total_tokens} "
                f"steps={m.avg_execution_steps:.2f}"
            )


if __name__ == "__main__":
    main()
