"""任务 4 的预算、披露边界与公平控制测试。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import Agent
from core.schemas import RetrievalResult, Skill, Task
from core.token_utils import estimate_tokens
from evaluation.run_benchmark import run_benchmark
from organization.flat import FlatOrganizer
from organization.hierarchical import HierarchicalOrganizer
from retrieval.bm25 import BM25Retriever


class FixedLLM:
    def __init__(self, skill_name: str, arguments: dict | None = None):
        self.skill_name = skill_name
        self.arguments = arguments or {}
        self.provider = "scripted"
        self.model = "scripted"
        self.temperature = 0.0
        self.thinking = None
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def generate_json(self, messages):
        return {
            "plan": [{
                "skill_name": self.skill_name,
                "arguments": self.arguments,
            }]
        }

    def generate(self, messages):
        return "not used"


def _skills() -> list[Skill]:
    return [
        Skill(
            id="s1",
            name="first_skill",
            brief_description="处理第一类输入",
            detailed_description="第一技能完整边界。" * 12,
            category="demo",
            parameters={"type": "object", "properties": {}},
        ),
        Skill(
            id="s2",
            name="second_skill",
            brief_description="处理第二类输入",
            detailed_description="第二技能不可被部分披露。" * 12,
            category="demo",
            parameters={"type": "object", "properties": {}},
        ),
    ]


def test_flat_budget_keeps_whole_skill_blocks_and_audit_ids():
    skills = _skills()
    first_only = FlatOrganizer().organize_context([skills[0]])
    budget = first_only.token_count
    organized = FlatOrganizer().organize_context(
        skills,
        context_budget_tokens=budget,
    )

    assert organized.token_count <= budget
    assert organized.exposed_skill_ids == ["s1"]
    assert organized.detailed_skill_ids == ["s1"]
    assert organized.truncated_skill_ids == ["s2"]
    assert skills[0].detailed_description in organized.text
    assert skills[1].name not in organized.text
    assert skills[1].detailed_description not in organized.text


def test_hierarchical_budget_and_disclosure_are_auditable():
    skills = _skills()
    organized = HierarchicalOrganizer(detail_top_k=1).organize_context(
        skills,
        context_budget_tokens=800,
    )

    assert organized.token_count <= 800
    assert organized.exposed_skill_ids == ["s1", "s2"]
    assert organized.detailed_skill_ids == ["s1"]
    assert skills[0].detailed_description in organized.text
    assert skills[1].detailed_description not in organized.text


def test_agent_cannot_execute_a_skill_that_was_not_exposed():
    skills = _skills()
    budget = FlatOrganizer().organize_context([skills[0]]).token_count
    agent = Agent(
        llm=FixedLLM("second_skill"),
        organizer=FlatOrganizer(),
        skill_handlers={"second_skill": lambda: "should not run"},
        enable_reflection=False,
    )
    result = agent.run(
        Task("t", "use second", ["s2"], ["s2"]),
        RetrievalResult("use second", skills, [1.0, 0.5]),
        context_budget_tokens=budget,
    )

    assert result["exposed_skill_ids"] == ["s1"]
    assert result["selected_skill_ids"] == []
    assert result["success"] is False


def test_benchmark_consumes_fixed_candidates_and_task_budget():
    skills = [
        {
            "id": "sa",
            "name": "skill_a",
            "brief_description": "alpha only",
            "detailed_description": "alpha handler",
        },
        {
            "id": "sb",
            "name": "skill_b",
            "brief_description": "beta only",
            "detailed_description": "beta handler",
        },
    ]
    tasks = [{
        "id": "t1",
        "instruction": "alpha request",
        "expected_skills": ["sb"],
        "expected_skill_sequence": ["sb"],
        "ground_truth": "done",
        "metadata": {
            "candidate_skill_ids": ["sb", "sa"],
            "slice": {
                "candidate_count": 2,
                "target_gold_rank": 1,
                "context_budget_tokens": 80,
            },
        },
    }]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        skills_path = root / "skills.json"
        tasks_path = root / "tasks.json"
        skills_path.write_text(json.dumps(skills), encoding="utf-8")
        tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
        common = {
            "skills_path": skills_path,
            "tasks_path": tasks_path,
            "retriever": BM25Retriever(text_level="brief"),
            "llm": FixedLLM("skill_b"),
            "skill_handlers": {"skill_b": lambda: "done"},
            "retrieval_ks": (1, 2),
            "enable_reflection": False,
            "use_task_candidate_fixtures": True,
            "use_task_context_budget": True,
        }
        flat = run_benchmark(organizer=FlatOrganizer(), **common)
        common["llm"] = FixedLLM("skill_b")
        hierarchical = run_benchmark(
            organizer=HierarchicalOrganizer(detail_top_k=1),
            **common,
        )

    for result in (flat, hierarchical):
        task_result = result["per_task"][0]
        assert task_result["retrieved_skill_ids"] == ["sb", "sa"]
        assert task_result["context_budget_tokens"] == 80
        assert task_result["skill_context_tokens"] <= 80
        assert result["config"]["candidate_source"] == "task_fixture"
        assert result["config"]["context_budget_source"] == "task_slice"
    assert flat["per_task"][0]["raw_bm25_skill_ids"][0] == "sa"


if __name__ == "__main__":
    test_flat_budget_keeps_whole_skill_blocks_and_audit_ids()
    test_hierarchical_budget_and_disclosure_are_auditable()
    test_agent_cannot_execute_a_skill_that_was_not_exposed()
    test_benchmark_consumes_fixed_candidates_and_task_budget()
    print("task4 experiment control tests passed")
