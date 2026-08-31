"""benchmark_v02 研究问题覆盖、排名、graph 与执行闭环测试。"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_skills, load_tasks
from evaluation.run_benchmark import run_benchmark
from execution.handlers import create_default_skill_registry
from organization.graph import GraphOrganizer
from organization.hierarchical import HierarchicalOrganizer
from retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_v02"


PLANS = {
    "v2body01": [("create_text_file", {"path": "$input.path", "content": "$input.content"})],
    "v2body03": [("append_text_file", {"path": "$input.path", "content": "$input.content"})],
    "v2body04": [("merge_json_objects", {"left": "$input.left", "right": "$input.right", "conflict_policy": "$input.conflict_policy"})],
    "v2body06": [("aggregate_sqlite_query", {"query": "$input.query"})],
    "v2rank01": [("update_sqlite_rows", {"table": "$input.table", "where": "$input.where", "changes": "$input.changes"})],
    "v2rank02": [("aggregate_number_list", {"values": "$input.values", "operation": "$input.operation"})],
    "v2graph01": [
        ("copy_file_preserve_source", {"source": "$input.source", "destination": "$input.destination"}),
        ("append_text_file", {"path": "$last_output", "content": "$input.append_content"}),
    ],
    "v2graph02": [
        ("create_text_file", {"path": "$input.path", "content": "$input.initial_content"}),
        ("patch_json_file", {"path": "$last_output", "patch": "$input.patch"}),
    ],
    "v2alt02": [("delete_sqlite_rows", {"table": "$input.table", "where": "$input.where"})],
    "v2schema01": [("merge_json_objects", {"left": "$input.left", "right": "$input.right", "conflict_policy": "$input.conflict_policy"})],
}


class ScriptedLLM:
    def __init__(self):
        self.provider = "scripted"
        self.model = "benchmark-v02-scripted"
        self.temperature = 0.0
        self.thinking = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tasks = {task.id: task for task in load_tasks(DATA / "tasks.jsonl")}

    def generate_json(self, messages):
        prompt = messages[-1]["content"]
        task_id = next(
            task_id for task_id, task in self.tasks.items()
            if f"任务：{task.instruction}\n" in prompt
        )
        return {"plan": [
            {"skill_name": name, "arguments": arguments}
            for name, arguments in PLANS[task_id]
        ]}

    def generate(self, messages):
        return "not used"


def _load():
    return load_skills(DATA / "skills.jsonl"), load_tasks(DATA / "tasks.jsonl")


def test_v02_has_required_size_splits_questions_and_private_verifiers():
    skills, tasks = _load()
    splits = Counter(task.metadata["split"] for task in tasks)
    questions = Counter(
        question
        for task in tasks
        for question in task.metadata["research_questions"]
    )

    assert len(skills) == 24
    assert len(tasks) == 20
    assert splits == Counter({"dev": 10, "test": 10})
    assert set(questions) == {
        "body_disambiguation",
        "low_initial_rank",
        "multi_skill_graph",
        "prerequisite_completion",
        "dataflow",
        "alternative_conflict",
        "schema_repair",
    }
    assert questions["body_disambiguation"] >= 6
    assert questions["schema_repair"] >= 2
    assert all(task.inputs for task in tasks)
    assert all(task.evaluation is not None for task in tasks)
    assert all(task.ground_truth is None for task in tasks)


def test_templates_are_split_isolated_and_raw_ranks_are_reproducible():
    skills, tasks = _load()
    template_splits: dict[str, set[str]] = defaultdict(set)
    retrievers = {
        level: BM25Retriever(text_level=level)
        for level in ("brief", "detailed", "all")
    }
    for retriever in retrievers.values():
        retriever.index(skills)

    for task in tasks:
        template_splits[task.metadata["template_id"]].add(task.metadata["split"])
        for level, retriever in retrievers.items():
            ranking = retriever.retrieve(task.instruction, len(skills)).ranked_ids()
            recorded = task.metadata["raw_ranks"][level]
            assert recorded == {
                skill_id: ranking.index(skill_id) + 1
                for skill_id in task.expected_skills
            }
    assert all(len(splits) == 1 for splits in template_splits.values())


def test_body_cases_create_retrieval_pressure_without_using_test_metrics():
    _, tasks = _load()
    dev_body = [
        task for task in tasks
        if task.metadata["split"] == "dev"
        and "body_disambiguation" in task.metadata["research_questions"]
    ]
    improved = 0
    for task in dev_body:
        brief_rank = min(task.metadata["raw_ranks"]["brief"].values())
        detailed_rank = min(task.metadata["raw_ranks"]["detailed"].values())
        all_rank = min(task.metadata["raw_ranks"]["all"].values())
        improved += int(detailed_rank < brief_rank or all_rank < brief_rank)
    assert len(dev_body) == 4
    assert improved >= 3


def test_typed_edges_separate_hard_expansion_from_soft_relations():
    skills, tasks = _load()
    by_id = {task.id: task for task in tasks}
    for task_id in ("v2graph01", "v2graph02"):
        task = by_id[task_id]
        omitted = task.metadata["intentionally_omitted_skill_ids"]
        edge_types = {edge["type"] for edge in task.metadata["graph_edges"]}
        assert omitted
        assert not set(omitted) & set(task.metadata["candidate_skill_ids"])
        assert edge_types == {"prerequisite", "dataflow"}
        organized = GraphOrganizer(
            catalog=skills,
            max_additional_skills=2,
        ).organize_context(
            [skill for skill in skills if skill.id in task.metadata["candidate_skill_ids"]],
            task,
            context_budget_tokens=task.metadata["context_budget_tokens"],
        )
        assert set(omitted) <= set(organized.added_skill_ids)

    for task_id in ("v2rank01", "v2alt01", "v2alt02"):
        task = by_id[task_id]
        assert {edge["type"] for edge in task.metadata["graph_edges"]} == {
            "alternative", "conflict",
        }


def test_all_dev_tasks_run_through_real_handlers_and_verifiers():
    skills, tasks = _load()
    dev_ids = {task.id for task in tasks if task.metadata["split"] == "dev"}
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=GraphOrganizer(catalog=skills, max_additional_skills=2),
        llm=ScriptedLLM(),
        skill_registry=create_default_skill_registry(),
        environment_fixtures_path=DATA / "environment_fixtures.json",
        task_ids=dev_ids,
        retrieval_ks=(1, 5, 10, 24),
        enable_reflection=False,
        use_task_candidate_fixtures=True,
        use_task_context_budget=True,
        run_id="benchmark-v02-scripted-dev",
    )

    assert result["run_info"]["task_count"] == 10
    assert result["metrics"]["agent"]["skill_selection_f1"] == 1.0
    assert result["metrics"]["agent"]["sequence_accuracy"] == 1.0
    assert result["metrics"]["agent"]["task_success_rate"] == 1.0
    graph_rows = {
        row["task_id"]: row for row in result["per_task"]
        if row["task_id"].startswith("v2graph")
    }
    assert graph_rows["v2graph01"]["added_skill_ids"] == ["sfile_copy"]
    assert graph_rows["v2graph02"]["added_skill_ids"] == ["sfile_create"]


def test_gold_augmented_candidates_are_explicit_and_do_not_inflate_retrieval_metrics():
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=HierarchicalOrganizer(detail_top_k=3),
        llm=ScriptedLLM(),
        skill_registry=create_default_skill_registry(),
        environment_fixtures_path=DATA / "environment_fixtures.json",
        task_ids=["v2body03"],
        top_k=10,
        retrieval_ks=(1, 5, 10, 20),
        enable_reflection=False,
        use_task_candidate_fixtures=False,
        ensure_gold_in_retrieval=True,
        use_task_context_budget=True,
        run_id="benchmark-v02-gold-augmentation-diagnostic",
    )
    row = result["per_task"][0]

    assert result["config"]["candidate_source"] == (
        "retriever_top_k_gold_augmented"
    )
    assert row["gold_augmented_skill_ids"] == ["sfile_append"]
    assert "sfile_append" in row["retrieved_skill_ids"]
    assert row["retrieval_metrics"]["recall@10"] == 0.0
    assert row["task_success"] is True


if __name__ == "__main__":
    test_v02_has_required_size_splits_questions_and_private_verifiers()
    test_templates_are_split_isolated_and_raw_ranks_are_reproducible()
    test_body_cases_create_retrieval_pressure_without_using_test_metrics()
    test_typed_edges_separate_hard_expansion_from_soft_relations()
    test_all_dev_tasks_run_through_real_handlers_and_verifiers()
    test_gold_augmented_candidates_are_explicit_and_do_not_inflate_retrieval_metrics()
    print("benchmark v02 tests passed")
