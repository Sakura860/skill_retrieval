"""Pinned official Graph-of-Skills reverse-PPR adapter tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_skills, load_tasks
from organization.gos_official import (
    OFFICIAL_COMMIT,
    OFFICIAL_QUERY_SHA256,
    OfficialGoSReversePPROrganizer,
    load_official_runtime,
)

DATA = ROOT / "data" / "benchmark_v02"
UPSTREAM = ROOT.parent / "tmp" / "graph-of-skills-upstream-20260903"
RESULT = ROOT / "results" / "gos_structural_baseline_dev_20260903.json"


def test_pinned_official_runtime_and_prerequisite_recovery() -> None:
    if not UPSTREAM.is_dir():
        print("Graph-of-Skills upstream checkout absent; integration test skipped")
        return
    runtime = load_official_runtime(UPSTREAM)
    assert callable(runtime.build_transition)
    assert callable(runtime.personalized_pagerank)
    assert len(OFFICIAL_COMMIT) == 40
    assert len(OFFICIAL_QUERY_SHA256) == 64

    skills = load_skills(DATA / "skills.jsonl")
    by_id = {skill.id: skill for skill in skills}
    tasks = {task.id: task for task in load_tasks(DATA / "tasks.jsonl")}
    expected = {
        "v2graph01": "sfile_copy",
        "v2graph02": "sfile_create",
    }
    for task_id, prerequisite in expected.items():
        task = tasks[task_id]
        seeds = [by_id[skill_id] for skill_id in task.metadata["candidate_skill_ids"]]
        organizer = OfficialGoSReversePPROrganizer(
            skills,
            UPSTREAM,
            max_additional_skills=2,
            include_dataflow=False,
        )
        result = organizer.organize_context(seeds, task, 3200)
        assert prerequisite in result.added_skill_ids
        assert len(result.added_skill_ids) == 2
        assert prerequisite in result.exposed_skill_ids
        assert organizer.provenance()["full_hybrid_pipeline"] is False


def test_completed_matched_result_keeps_claim_boundary() -> None:
    if not RESULT.is_file():
        print("GoS matched result absent; completed-result test skipped")
        return
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = result["protocol_snapshot"]
    assert protocol["scope"]["split"] == "dev_only"
    assert protocol["scope"]["old_task5_confirmation_used"] is False
    assert result["official_provenance"]["full_hybrid_pipeline"] is False
    typed = result["aggregates"]["typed_prerequisite_completion"]
    gos = result["aggregates"]["gos_reverse_ppr_prerequisite"]
    gos_flow = result["aggregates"][
        "gos_reverse_ppr_prerequisite_plus_dataflow"
    ]
    assert typed["mean_missing_prerequisite_recall"] == 1.0
    assert typed["mean_irrelevant_additions"] == 0.0
    assert typed["task_success_rate"] == 1.0
    assert gos["mean_missing_prerequisite_recall"] == 1.0
    assert gos["mean_irrelevant_additions"] == 1.0
    assert gos["task_success_rate"] == 1.0
    assert gos_flow["mean_missing_prerequisite_recall"] == 1.0
    assert gos_flow["mean_irrelevant_additions"] == 1.0


if __name__ == "__main__":
    test_pinned_official_runtime_and_prerequisite_recovery()
    test_completed_matched_result_keeps_claim_boundary()
    print("official GoS adapter tests passed")
