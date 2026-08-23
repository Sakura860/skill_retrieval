"""24 条真实 handler + verifier 的确定性端到端基线。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_tasks
from evaluation.run_benchmark import run_benchmark
from execution.handlers import create_default_skill_registry
from execution.registry import SkillRegistry
from organization.flat import FlatOrganizer
from retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_v01"


PLANS = {
    "tcalc01": [("evaluate_integer_expression", {"expression": "(18 + 7) * 4"})],
    "tcalc02": [("evaluate_decimal_expression", {"expression": "12.5 / 4"})],
    "tcalc03": [("calculate_percentage_change", {"old_value": 80, "new_value": 100})],
    "tcalc04": [("solve_linear_equation", {"a": 3, "b": 6, "c": 21})],
    "tcalc05": [("convert_measurement_units", {"value": 2.5, "from_unit": "千米", "to_unit": "米"})],
    "tcalc06": [("aggregate_number_list", {"values": [5, 7, 11, 19], "operation": "sum"})],
    "tcalc07": [
        ("evaluate_decimal_expression", {"expression": "12.5 + 7.5"}),
        ("calculate_percentage_change", {"old_value": 16, "new_value": "$last_output"}),
    ],
    "tcalc08": [
        ("convert_measurement_units", {"value": 3, "from_unit": "千米", "to_unit": "米"}),
        ("evaluate_integer_expression", {"expression": "3000 + 250"}),
    ],
    "tjson01": [("extract_json_fields", {"data": {"name": "Ada", "age": 37, "city": "London"}, "fields": ["name", "age"]})],
    "tjson02": [("rename_json_keys", {"data": {"user_name": "Ada", "age": 37}, "mapping": {"user_name": "name"}})],
    "tjson03": [("filter_json_records", {"records": [{"name": "Ada", "active": True}, {"name": "Bob", "active": False}], "field": "active", "equals": True})],
    "tjson04": [("merge_json_objects", {"left": {"theme": "light", "lang": "zh"}, "right": {"theme": "dark"}, "conflict_policy": "right"})],
    "tjson05": [("sort_json_records", {"records": [{"name": "A", "score": 8}, {"name": "B", "score": 10}], "field": "score", "descending": True})],
    "tjson06": [("convert_csv_to_json", {"csv_text": "name,age\nAda,37\nBob,16"})],
    "tjson07": [
        ("extract_json_fields", {"data": {"user_name": "Ada", "age": 37, "city": "London"}, "fields": ["user_name", "age"]}),
        ("rename_json_keys", {"data": "$last_output", "mapping": {"user_name": "name"}}),
    ],
    "tjson08": [
        ("filter_json_records", {"records": [{"name": "A", "active": True, "score": 8}, {"name": "B", "active": False, "score": 10}, {"name": "C", "active": True, "score": 9}], "field": "active", "equals": True}),
        ("sort_json_records", {"records": "$last_output", "field": "score", "descending": True}),
    ],
    "tfile01": [("create_text_file", {"path": "report.txt", "content": "hello"})],
    "tfile02": [("overwrite_text_file", {"path": "report.txt", "content": "final"})],
    "tfile03": [("append_text_file", {"path": "log.txt", "content": "\ndone"})],
    "tfile04": [("copy_file_preserve_source", {"source": "source.txt", "destination": "backup.txt"})],
    "tfile05": [("move_file", {"source": "draft.txt", "destination": "final.txt"})],
    "tfile06": [("patch_json_file", {"path": "settings.json", "patch": {"theme": "dark"}})],
    "tfile07": [
        ("copy_file_preserve_source", {"source": "source.txt", "destination": "backup.txt"}),
        ("append_text_file", {"path": "backup.txt", "content": "\nextra"}),
    ],
    "tdb01": [("select_sqlite_rows", {"query": "SELECT name FROM users WHERE age >= 18 ORDER BY name"})],
    "tdb02": [("insert_sqlite_row", {"table": "users", "values": {"id": 4, "name": "Dora", "age": 22, "active": 1}})],
    "tdb03": [("update_sqlite_rows", {"table": "users", "where": {"name": "Bob"}, "changes": {"active": 1}})],
    "tdb04": [("delete_sqlite_rows", {"table": "users", "where": {"active": 0}})],
    "tdb05": [("aggregate_sqlite_query", {"query": "SELECT COUNT(*) FROM users WHERE age >= 18"})],
    "tdb06": [("join_sqlite_tables", {"query": "SELECT o.id, u.name, o.amount FROM orders o JOIN users u ON u.id=o.user_id ORDER BY o.id"})],
    "tdb07": [
        ("insert_sqlite_row", {"table": "users", "values": {"id": 4, "name": "Dora", "age": 17, "active": 0}}),
        ("update_sqlite_rows", {"table": "users", "where": {"name": "Dora"}, "changes": {"active": 1}}),
        ("select_sqlite_rows", {"query": "SELECT id, name, age, active FROM users WHERE name='Dora'"}),
    ],
}


class ScriptedLLM:
    def __init__(self, plans_by_instruction: dict[str, list[tuple[str, dict]]]):
        self.provider = "scripted"
        self.model = "scripted"
        self.temperature = 0.0
        self.thinking = None
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.plans_by_instruction = plans_by_instruction

    def generate_json(self, messages):
        prompt = messages[-1]["content"]
        for instruction, steps in self.plans_by_instruction.items():
            if f"任务：{instruction}\n" in prompt:
                return {
                    "plan": [
                        {"skill_name": name, "arguments": arguments}
                        for name, arguments in steps
                    ]
                }
        return {"plan": []}

    def generate(self, messages):
        return "not used"


def _scripted_llm(task_ids: set[str]) -> ScriptedLLM:
    tasks = load_tasks(DATA / "tasks.jsonl")
    by_id = {task.id: task for task in tasks}
    return ScriptedLLM({
        by_id[task_id].instruction: PLANS[task_id]
        for task_id in task_ids
    })


def test_30_tasks_run_through_real_handlers_and_verifiers():
    task_ids = set(PLANS)
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=FlatOrganizer(),
        llm=_scripted_llm(task_ids),
        skill_registry=create_default_skill_registry(),
        environment_fixtures_path=DATA / "environment_fixtures.json",
        task_ids=task_ids,
        top_k=24,
        retrieval_ks=(1, 5, 10, 24),
        enable_reflection=False,
        run_id="real-handlers-30",
    )

    assert result["run_info"]["task_count"] == 30
    assert result["run_info"]["scored_task_count"] == 30
    assert result["run_info"]["unscored_task_count"] == 0
    assert result["metrics"]["agent"]["skill_selection_f1"] == 1.0
    assert result["metrics"]["agent"]["sequence_accuracy"] == 1.0
    assert result["metrics"]["agent"]["task_success_rate"] == 1.0
    assert all(row["execution_success"] is True for row in result["per_task"])
    assert all(row["task_success"] is True for row in result["per_task"])


def test_handler_success_without_required_effect_is_task_failure():
    task_id = "tfile01"
    fake_registry = SkillRegistry()
    fake_registry.register(
        "create_text_file",
        lambda environment, path, content: path,
        {"file:write"},
    )
    result = run_benchmark(
        DATA / "skills.jsonl",
        DATA / "tasks.jsonl",
        retriever=BM25Retriever(text_level="brief"),
        organizer=FlatOrganizer(),
        llm=_scripted_llm({task_id}),
        skill_registry=fake_registry,
        task_ids={task_id},
        top_k=24,
        retrieval_ks=(1, 24),
        enable_reflection=False,
        run_id="fake-success",
    )

    row = result["per_task"][0]
    assert row["execution_success"] is True
    assert row["task_success"] is False
    assert row["failure_reason"] == "file_state_mismatch"
    assert result["metrics"]["agent"]["task_success_rate"] == 0.0


if __name__ == "__main__":
    test_30_tasks_run_through_real_handlers_and_verifiers()
    test_handler_success_without_required_effect_is_task_failure()
    print("real handler end-to-end tests passed")
