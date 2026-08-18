"""项目演示入口。"""
from __future__ import annotations

from agent.agent import Agent
from core.config import PROJECT_ROOT, load_config
from core.llm import LLM
from data.loader import load_skills, load_tasks
from evaluation.task_metrics import evaluate_agent
from organization.flat import FlatOrganizer
from organization.graph import GraphOrganizer
from organization.hierarchical import HierarchicalOrganizer
from retrieval.bm25 import BM25Retriever
from retrieval.embedding import EmbeddingRetriever
from retrieval.evaluator import evaluate_retrieval
from retrieval.multilevel import MultiLevelRetriever


def build_retriever(config: dict):
    retrieval_config = config["retrieval"]
    method = retrieval_config["method"]
    if method == "bm25":
        return BM25Retriever()
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
    skills = load_skills(PROJECT_ROOT / "data" / "skills")
    tasks = load_tasks(PROJECT_ROOT / "data" / "tasks")
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
    agent_config = config["agent"]
    agent = Agent(
        llm=llm,
        organizer=organizer,
        max_steps=agent_config["max_steps"],
        enable_reflection=agent_config["enable_reflection"],
    )

    retrieval_metrics = evaluate_retrieval(retriever, tasks, skills)
    retriever.index(skills)
    top_k = config["retrieval"]["top_k"]

    def run_task(task):
        result = retriever.retrieve(task.instruction, top_k=top_k)
        return agent.run(task, result)

    task_metrics = evaluate_agent(run_task, tasks)

    print(f"加载 Skill {len(skills)} 个，任务 {len(tasks)} 个")
    print(f"LLM: {llm.provider} / {llm.model}")
    print(f"思考模式: {llm.thinking or 'not-applicable'}")
    print(f"检索器: {type(retriever).__name__}")
    print(f"组织器: {type(organizer).__name__}")
    print("\n检索指标")
    for name, value in retrieval_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nAgent 指标")
    for name, value in task_metrics.to_dict().items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {name}: {formatted}")


if __name__ == "__main__":
    main()
