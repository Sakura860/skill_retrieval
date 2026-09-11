"""Unseen-family HTTP benchmark data, runtime, disclosure, and verifier tests."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_skills, load_tasks
from evaluation.run_benchmark import run_benchmark
from execution.environment import TaskEnvironment, load_environment_fixtures
from execution.handlers import create_default_skill_registry
from organization.hierarchical import HierarchicalOrganizer
from retrieval.bm25 import BM25Retriever

DATA = ROOT / "data" / "benchmark_http_v01"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PLANS = {
    "thttpd01": [("http_get_json", {"path": "$input.path"})],
    "thttpd02": [("http_head_status", {"path": "$input.path"})],
    "thttpd03": [
        ("http_issue_token", {"scope": "$input.scope"}),
        ("http_get_bearer_json", {"path": "$input.path", "token": "$last_output"}),
    ],
    "thttpd04": [("http_post_json", {
        "path": "$input.path",
        "payload": "$input.payload",
        "idempotency_key": "$input.idempotency_key",
        "accepted_statuses": [201],
        "on_unaccepted": "return_body",
    })],
    "thttpd05": [("http_put_json", {
        "path": "$input.path", "payload": "$input.payload",
    })],
    "thttpd06": [("http_get_json", {
        "path": "$input.path",
        "accepted_statuses": [200],
        "on_unaccepted": "return_body",
    })],
    "thttpc01": [("http_get_json", {"path": "$input.path"})],
    "thttpc02": [("http_head_status", {"path": "$input.path"})],
    "thttpc03": [
        ("http_issue_token", {"scope": "$input.scope"}),
        ("http_get_bearer_json", {"path": "$input.path", "token": "$last_output"}),
    ],
    "thttpc04": [("http_post_json", {
        "path": "$input.path",
        "payload": "$input.payload",
        "idempotency_key": "$input.idempotency_key",
        "accepted_statuses": [201],
        "on_unaccepted": "return_body",
    })],
    "thttpc05": [("http_patch_json", {
        "path": "$input.path", "patch": "$input.patch",
    })],
    "thttpc06": [("http_get_json", {
        "path": "$input.path",
        "accepted_statuses": [200],
        "on_unaccepted": "return_body",
    })],
}


class ScriptedHTTPPlanner:
    def __init__(self):
        self.provider = "scripted"
        self.model = "scripted-http"
        self.temperature = 0.0
        self.thinking = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tasks = load_tasks(DATA / "tasks.jsonl")
        self.by_instruction = {task.instruction: task for task in tasks}

    def generate_json(self, messages):
        prompt = messages[-1]["content"]
        task = next(
            task for instruction, task in self.by_instruction.items()
            if f"任务：{instruction}\n" in prompt
        )
        if "此阶段不要生成参数" in prompt:
            return {"skill_ids": task.expected_skills}
        return {
            "plan": [
                {"skill_name": name, "arguments": arguments, "reason": "fixture"}
                for name, arguments in PLANS[task.id]
            ]
        }

    def generate(self, messages):
        return "not used"


def test_dataset_is_isolated_and_complete() -> None:
    skills = load_skills(DATA / "skills.jsonl")
    tasks = load_tasks(DATA / "tasks.jsonl")
    fixtures = load_environment_fixtures(DATA / "environment_fixtures.json")
    assert len(skills) == 9
    assert len(tasks) == 12
    assert len(fixtures) == 12
    assert {task.metadata["split"] for task in tasks} == {"dev", "confirmation"}
    dev = [task for task in tasks if task.metadata["split"] == "dev"]
    confirmation = [task for task in tasks if task.metadata["split"] == "confirmation"]
    assert len(dev) == len(confirmation) == 6
    assert {task.metadata["template_id"] for task in dev}.isdisjoint(
        {task.metadata["template_id"] for task in confirmation}
    )
    assert all(task.evaluation.verifier_type == "http_state" for task in tasks)
    assert all(task.metadata["family"] == "local_http_api" for task in tasks)


def test_frozen_protocol_policy_and_single_confirmation_registry() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "http_transfer_protocol_20260903.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "frozen"
    assert file_sha256(ROOT / protocol["frozen_policy"]["path"]) == (
        protocol["frozen_policy"]["sha256"]
    )
    assert file_sha256(DATA / "skills.jsonl") == protocol["dataset"]["skills_sha256"]
    assert file_sha256(DATA / "tasks.jsonl") == protocol["dataset"]["tasks_sha256"]
    assert file_sha256(DATA / "environment_fixtures.json") == (
        protocol["dataset"]["environment_fixtures_sha256"]
    )
    registry = json.loads(
        (ROOT / "results" / "http_transfer_confirmation_registry.json").read_text(
            encoding="utf-8"
        )
    )
    runs = [
        row for row in registry["runs"]
        if row["protocol_id"] == protocol["protocol_id"]
    ]
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    result_path = Path(runs[0]["output_path"])
    assert file_sha256(result_path) == runs[0]["result_sha256"]


def test_http_runtime_rejects_nonlocal_url() -> None:
    task = load_tasks(DATA / "tasks.jsonl")[0]
    fixtures = load_environment_fixtures(DATA / "environment_fixtures.json")
    with TaskEnvironment(task, fixtures) as environment:
        try:
            environment.request_http_json("GET", "https://example.com/")
        except PermissionError:
            pass
        else:
            raise AssertionError("absolute external URL should be rejected")


def test_all_tasks_pass_real_http_handlers_and_http_verifier() -> None:
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=HierarchicalOrganizer(detail_top_k=0),
        llm=ScriptedHTTPPlanner(),
        skill_registry=create_default_skill_registry(),
        environment_fixtures_path=DATA / "environment_fixtures.json",
        top_k=9,
        retrieval_ks=(1, 3, 5, 9),
        enable_reflection=False,
        planner_mode="two_stage",
        planner_disclosure_level="adaptive_signals",
        max_argument_repairs=1,
        run_id="http-scripted-e2e",
    )
    assert result["run_info"]["task_count"] == 12
    assert result["metrics"]["agent"]["task_success_rate"] == 1.0
    assert result["metrics"]["agent"]["skill_selection_f1"] == 1.0
    assert result["metrics"]["agent"]["sequence_accuracy"] == 1.0
    rows = {row["task_id"]: row for row in result["per_task"]}
    assert rows["thttpd01"]["resolved_disclosure_level"] == "brief"
    assert rows["thttpd03"]["resolved_disclosure_level"] == "schema"
    assert rows["thttpd04"]["resolved_disclosure_level"] == "full"
    assert rows["thttpd06"]["resolved_disclosure_level"] == "full"
    assert rows["thttpc01"]["resolved_disclosure_level"] == "brief"
    assert rows["thttpc03"]["resolved_disclosure_level"] == "schema"
    assert rows["thttpc04"]["resolved_disclosure_level"] == "full"
    assert rows["thttpc06"]["resolved_disclosure_level"] == "full"
    for row in rows.values():
        assert row["verifier_type"] == "http_state"
        assert row["task_success"] is True


if __name__ == "__main__":
    test_dataset_is_isolated_and_complete()
    test_frozen_protocol_policy_and_single_confirmation_registry()
    test_http_runtime_rejects_nonlocal_url()
    test_all_tasks_pass_real_http_handlers_and_http_verifier()
    print("HTTP benchmark tests passed")
