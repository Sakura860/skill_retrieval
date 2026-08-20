"""确定性任务完成度 verifier 测试。"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.schemas import Task, TaskEvaluationConfig
from evaluation.task_metrics import evaluate_agent
from evaluation.verifiers import verify_task


def _result(answer="", *, success=True, initial_state=None, final_state=None):
    return {
        "success": success,
        "answer": answer,
        "trajectory": [{"step": "action", "success": success}],
        "initial_state": initial_state,
        "final_state": final_state,
        "selected_skill_ids": ["s1"],
    }


def test_task_evaluation_dict_is_normalized():
    task = Task(
        id="t1",
        instruction="calculate",
        evaluation={
            "verifier_type": "exact_match",
            "expected_output": 2,
            "tolerance": 0.01,
        },
    )

    assert isinstance(task.evaluation, TaskEvaluationConfig)
    assert task.evaluation.expected_output == 2


def test_exact_match_supports_tolerance():
    task = Task(
        "t1",
        "calculate",
        evaluation=TaskEvaluationConfig(
            verifier_type="exact_match",
            expected_output=3.14,
            tolerance=0.01,
        ),
    )

    verification = verify_task(task, _result("3.145"))

    assert verification is not None
    assert verification.passed is True
    assert verification.verifier_type == "exact_match"


def test_json_handler_success_but_missing_field_is_task_failure():
    task = Task(
        "t-json",
        "return user json",
        expected_skills=["s1"],
        evaluation=TaskEvaluationConfig(
            verifier_type="json_match",
            expected_output={"name": "Ada", "age": 37},
        ),
    )
    result = _result(json.dumps({"name": "Ada"}))

    metrics = evaluate_agent(lambda _: result, [task])

    row = metrics.per_task[0]
    assert row["execution_success"] is True
    assert row["task_success"] is False
    assert row["verifier_type"] == "json_match"
    assert row["failure_reason"] == "output_mismatch"
    assert metrics.task_success_rate == 0.0


def test_file_handler_success_but_wrong_file_is_task_failure():
    task = Task(
        "t-file",
        "write result",
        expected_skills=["s1"],
        evaluation=TaskEvaluationConfig(
            verifier_type="file_state",
            expected_state={
                "files": {
                    "result.json": {
                        "exists": True,
                        "json": {"total": 231},
                    }
                }
            },
            required_effects=["result.json"],
            forbidden_effects=["source.json"],
        ),
    )
    initial = {
        "files": {
            "source.json": {"exists": True, "content": '{"value": 1}'}
        }
    }
    final = {
        "files": {
            "source.json": {"exists": True, "content": '{"value": 2}'},
            "result.json": {"exists": True, "json": {"total": 231}},
        }
    }

    metrics = evaluate_agent(
        lambda _: _result("done", initial_state=initial, final_state=final),
        [task],
    )

    row = metrics.per_task[0]
    assert row["execution_success"] is True
    assert row["task_success"] is False
    assert row["failure_reason"] == "forbidden_effect_detected"
    assert sorted(row["verifier_details"]["effects"]) == [
        "result.json",
        "source.json",
    ]


def test_file_handler_success_but_wrong_content_is_task_failure():
    task = Task(
        "t-file-content",
        "write result",
        evaluation=TaskEvaluationConfig(
            verifier_type="file_state",
            expected_state={"files": {"result.txt": "correct"}},
        ),
    )

    verification = verify_task(
        task,
        _result(
            "done",
            initial_state={"files": {}},
            final_state={"files": {"result.txt": "wrong"}},
        ),
    )

    assert verification is not None
    assert verification.passed is False
    assert verification.failure_reason == "file_state_mismatch"


def test_sqlite_verifier_checks_database_state():
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "task.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("CREATE TABLE users(name TEXT, age INTEGER)")
            connection.executemany(
                "INSERT INTO users VALUES (?, ?)",
                [("Ada", 37), ("Bob", 16)],
            )
            connection.commit()

        task = Task(
            "t-sqlite",
            "find adults",
            evaluation=TaskEvaluationConfig(
                verifier_type="sqlite_state",
                expected_state={
                    "query": "SELECT name FROM users WHERE age >= ? ORDER BY name",
                    "params": [18],
                    "expected_rows": [["Ada"]],
                },
            ),
        )

        verification = verify_task(
            task,
            _result("queried", final_state={"database_path": str(database_path)}),
        )

        assert verification is not None
        assert verification.passed is True
        assert verification.actual == [["Ada"]]


def test_sqlite_wrong_result_is_rejected():
    task = Task(
        "t-sqlite-wrong",
        "find adults",
        evaluation=TaskEvaluationConfig(
            verifier_type="sqlite_state",
            expected_state={"expected_rows": [["Ada"]]},
        ),
    )

    verification = verify_task(
        task,
        _result("queried", final_state={"query_result": [["Bob"]]}),
    )

    assert verification is not None
    assert verification.passed is False
    assert verification.failure_reason == "sqlite_result_mismatch"


def test_unknown_verifier_is_scored_failure():
    task = Task(
        "t-unknown",
        "unknown",
        evaluation=TaskEvaluationConfig(
            verifier_type="does_not_exist",
            expected_output="ok",
        ),
    )

    metrics = evaluate_agent(lambda _: _result("ok"), [task])

    assert metrics.scored_task_count == 1
    assert metrics.per_task[0]["task_success"] is False
    assert metrics.per_task[0]["failure_reason"] == "unknown_verifier_type"


if __name__ == "__main__":
    test_task_evaluation_dict_is_normalized()
    test_exact_match_supports_tolerance()
    test_json_handler_success_but_missing_field_is_task_failure()
    test_file_handler_success_but_wrong_file_is_task_failure()
    test_file_handler_success_but_wrong_content_is_task_failure()
    test_sqlite_verifier_checks_database_state()
    test_sqlite_wrong_result_is_rejected()
    test_unknown_verifier_is_scored_failure()
    print("verifier tests passed")
