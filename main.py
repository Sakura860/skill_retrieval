"""项目演示入口。"""
from __future__ import annotations

import json

from core.config import PROJECT_ROOT, load_config
from core.llm import LLM
from evaluation.run_benchmark import run_benchmark
from execution.handlers import create_default_skill_registry
from organization.flat import FlatOrganizer
from organization.graph import GraphOrganizer
from organization.hierarchical import HierarchicalOrganizer
from retrieval.bm25 import BM25Retriever
from retrieval.embedding import EmbeddingRetriever
from retrieval.multilevel import MultiLevelRetriever


def build_retriever(config: dict):
    retrieval_config = config["retrieval"]
    method = retrieval_config["method"]
    if method == "bm25":
        return BM25Retriever(
            text_level=retrieval_config.get("bm25_text_level", "brief")
        )
    if method == "embedding":
        return EmbeddingRetriever(retrieval_config["embedding_model"])
    if method == "multilevel":
        return MultiLevelRetriever(
            coarse_k=retrieval_config["coarse_k"],
            coarse_weight=retrieval_config["coarse_weight"],
        )
    if method == "model":
        from retrieval.model import RetrievalModel

        return RetrievalModel()
    raise ValueError(f"主入口暂不支持检索方法: {method}")


def build_organizer(config: dict):
    organization_config = config["organization"]
    strategy = organization_config["strategy"]
    if strategy == "flat":
        return FlatOrganizer()
    if strategy == "hierarchical":
        return HierarchicalOrganizer(organization_config["detail_top_k"])
    if strategy == "graph":
        return GraphOrganizer()
    raise ValueError(f"未知组织策略: {strategy}")


def main() -> None:
    config = load_config()
    retriever = build_retriever(config)
    organizer = build_organizer(config)

    llm_config = config["llm"]
    llm = LLM(
        provider=llm_config["provider"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        thinking=llm_config.get("thinking"),
        reasoning_effort=llm_config.get("reasoning_effort"),
    )
    top_k = config["retrieval"]["top_k"]
    agent_config = config["agent"]
    result = run_benchmark(
        PROJECT_ROOT / "data" / "skills",
        PROJECT_ROOT / "data" / "tasks",
        retriever=retriever,
        organizer=organizer,
        llm=llm,
        skill_registry=create_default_skill_registry(),
        top_k=top_k,
        retrieval_ks=tuple(config["evaluation"]["retrieval_ks"]),
        max_steps=agent_config["max_steps"],
        enable_reflection=agent_config["enable_reflection"],
        planner_mode=agent_config.get("planner_mode", "one_stage"),
        max_argument_repairs=agent_config.get("max_argument_repairs", 1),
        planner_disclosure_level=agent_config.get(
            "planner_disclosure_level", "full"
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
