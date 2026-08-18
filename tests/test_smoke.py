"""核心链路冒烟测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import Agent
from core.llm import LLM
from data.loader import load_skills, load_tasks
from organization.flat import FlatOrganizer
from retrieval.bm25 import BM25Retriever
from retrieval.evaluator import evaluate_retrieval
from retrieval.multilevel import MultiLevelRetriever

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end():
    skills = load_skills(ROOT / "data" / "skills")
    tasks = load_tasks(ROOT / "data" / "tasks")
    assert skills and tasks, "样例数据缺失"
    assert skills[0].brief_description
    assert skills[0].detailed_description

    retriever = BM25Retriever()
    retriever.index(skills)

    scores = evaluate_retrieval(retriever, tasks, skills)
    assert "recall@5" in scores

    multilevel = MultiLevelRetriever()
    multilevel.index(skills)
    assert multilevel.retrieve(tasks[0].instruction, top_k=3).skills

    agent = Agent(llm=LLM(provider="mock"), organizer=FlatOrganizer())
    res = agent.run(tasks[0], retriever.retrieve(tasks[0].instruction, top_k=5))
    assert "trajectory" in res and "success" in res
    assert "selected_skill_ids" in res
    assert "skill_context_tokens" in res
    assert "token_usage" in res


if __name__ == "__main__":
    test_end_to_end()
    print("smoke test passed")
