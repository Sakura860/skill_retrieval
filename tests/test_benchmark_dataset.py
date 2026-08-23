"""benchmark_v01 定向数据完整性与实验切片测试。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_skills, load_tasks
from organization.graph import GraphOrganizer
from retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_v01"
FAMILIES = {"calculation", "json_text", "file_operation", "sqlite"}


def _load():
    return (
        load_skills(DATA / "skills.jsonl"),
        load_tasks(DATA / "tasks.jsonl"),
        load_skills(DATA / "graph_skills.jsonl"),
        load_tasks(DATA / "graph_tasks.jsonl"),
    )


def test_main_skill_boundaries_are_complete():
    skills, _, _, _ = _load()
    family_counts = Counter(item.category for item in skills)

    assert len(skills) == 24
    assert family_counts == Counter({family: 6 for family in FAMILIES})
    for item in skills:
        assert item.brief_description
        assert "适用边界" in item.detailed_description
        assert item.parameters.get("type") == "object"
        assert item.returns
        assert any(example.startswith("正例：") for example in item.examples)
        assert any(example.startswith("反例：") for example in item.examples)


def test_main_tasks_cover_families_splits_steps_and_verifiers():
    _, tasks, _, _ = _load()
    families = Counter(task.metadata["family"] for task in tasks)
    splits = Counter(task.metadata["split"] for task in tasks)
    step_counts = Counter(task.metadata["step_count"] for task in tasks)
    verifier_types = {task.evaluation.verifier_type for task in tasks}

    assert len(tasks) == 34
    assert set(families) == FAMILIES
    assert min(families.values()) >= 8
    assert splits == Counter({"dev": 18, "test": 16})
    assert set(step_counts) == {1, 2, 3}
    assert verifier_types == {
        "exact_match",
        "json_match",
        "file_state",
        "sqlite_state",
    }
    assert all(task.ground_truth is None for task in tasks)


def test_candidate_fixtures_control_rank_and_cover_slice_grid():
    skills, tasks, _, _ = _load()
    skill_ids = {item.id for item in skills}
    observed_top_k = set()
    observed_rank = set()
    observed_budget = set()
    combinations = set()

    for task in tasks:
        metadata = task.metadata
        slice_info = metadata["slice"]
        candidates = metadata["candidate_skill_ids"]
        candidate_count = slice_info["candidate_count"]
        target_rank = slice_info["target_gold_rank"]
        primary = slice_info["primary_gold_skill_id"]

        assert metadata["candidate_generation"] == (
            "bm25_brief_then_gold_rank_control"
        )
        assert len(candidates) == candidate_count
        assert len(candidates) == len(set(candidates))
        assert set(candidates) <= skill_ids
        assert set(task.expected_skills) <= set(candidates)
        assert candidates[target_rank - 1] == primary
        assert primary == task.expected_skill_sequence[0]

        budget = slice_info["context_budget_tokens"]
        observed_top_k.add(candidate_count)
        observed_rank.add(target_rank)
        observed_budget.add(budget)
        combinations.add((candidate_count, target_rank, budget))

    assert observed_top_k == {5, 10, 20}
    assert observed_rank == {1, 3, 5}
    assert observed_budget == {800, 1200, 2000}
    assert len(combinations) == 27


def test_recorded_raw_bm25_ranks_match_current_retriever():
    skills, tasks, _, _ = _load()
    retriever = BM25Retriever(text_level="brief")
    retriever.index(skills)

    for task in tasks:
        raw_ranking = retriever.retrieve(
            task.instruction,
            top_k=len(skills),
        ).ranked_ids()
        primary = task.metadata["slice"]["primary_gold_skill_id"]
        recorded_rank = task.metadata["slice"]["raw_bm25_primary_rank"]
        assert raw_ranking.index(primary) + 1 == recorded_rank


def test_templates_do_not_cross_dev_test_and_variants_are_numbered():
    _, tasks, _, _ = _load()
    template_splits: dict[str, set[str]] = defaultdict(set)
    template_variants: dict[str, list[int]] = defaultdict(list)

    for task in tasks:
        template = task.metadata["template_id"]
        template_splits[template].add(task.metadata["split"])
        template_variants[template].append(task.metadata["variant_id"])

    assert all(len(splits) == 1 for splits in template_splits.values())
    repeated = 0
    for variants in template_variants.values():
        assert sorted(variants) == list(range(1, len(variants) + 1))
        repeated += int(len(variants) > 1)
    assert repeated == 4


def test_environment_fixtures_and_task_evaluations_are_complete():
    _, tasks, _, _ = _load()
    fixtures = json.loads(
        (DATA / "environment_fixtures.json").read_text(encoding="utf-8")
    )

    for task in tasks:
        assert task.evaluation is not None
        environment = task.metadata["environment_fixture"]
        if task.metadata["family"] == "file_operation":
            assert "initial_state" in environment
        if task.metadata["family"] == "sqlite":
            fixture_id = environment["sqlite_fixture_id"]
            assert fixture_id in fixtures
            assert fixtures[fixture_id]["type"] == "sqlite"
            assert fixtures[fixture_id]["setup_sql"]


def test_graph_diagnostics_are_isolated_and_dependency_consistent():
    _, _, graph_skills, graph_tasks = _load()
    skill_by_id = {item.id: item for item in graph_skills}
    diagnostic_types = {task.metadata["diagnostic_type"] for task in graph_tasks}

    assert len(graph_skills) == 12
    assert len(graph_tasks) == 10
    assert "missing_dependency" in diagnostic_types
    assert "cycle_detection" in diagnostic_types
    assert "branch_merge_dependency" in diagnostic_types
    assert "long_chain" in diagnostic_types

    for task in graph_tasks:
        candidates = task.metadata["candidate_skill_ids"]
        assert set(candidates) <= set(skill_by_id)
        assert set(task.expected_skills) <= set(candidates)
        if task.metadata["expected_graph_issue"] is None:
            positions = {
                skill_id: index
                for index, skill_id in enumerate(task.expected_skill_sequence)
            }
            for skill_id in task.expected_skill_sequence:
                for dependency in skill_by_id[skill_id].dependencies:
                    assert dependency in positions
                    assert positions[dependency] < positions[skill_id]

    missing_task = next(task for task in graph_tasks if task.id == "tg08")
    missing_context = GraphOrganizer().organize(
        [skill_by_id[item] for item in missing_task.metadata["candidate_skill_ids"]],
        missing_task,
    )
    assert "g_external_dictionary" in missing_context
    assert "缺少依赖" in missing_context

    cycle_task = next(task for task in graph_tasks if task.id == "tg09")
    cycle_context = GraphOrganizer().organize(
        [skill_by_id[item] for item in cycle_task.metadata["candidate_skill_ids"]],
        cycle_task,
    )
    assert "依赖环" in cycle_context
    assert "g_cycle_a" in cycle_context
    assert "g_cycle_b" in cycle_context


if __name__ == "__main__":
    test_main_skill_boundaries_are_complete()
    test_main_tasks_cover_families_splits_steps_and_verifiers()
    test_candidate_fixtures_control_rank_and_cover_slice_grid()
    test_recorded_raw_bm25_ranks_match_current_retriever()
    test_templates_do_not_cross_dev_test_and_variants_are_numbered()
    test_environment_fixtures_and_task_evaluations_are_complete()
    test_graph_diagnostics_are_isolated_and_dependency_consistent()
    print("benchmark dataset tests passed")
