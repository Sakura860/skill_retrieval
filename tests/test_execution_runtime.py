"""TaskEnvironment、SkillRegistry 与安全边界测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.schemas import Task
from execution.environment import TaskEnvironment
from execution.handlers import create_default_skill_registry


def _file_task() -> Task:
    return Task(
        id="runtime-file",
        instruction="append file",
        metadata={
            "environment_fixture": {
                "initial_state": {
                    "files": {
                        "input.txt": {"exists": True, "content": "start"}
                    }
                }
            }
        },
    )


def test_environment_setup_snapshot_and_teardown():
    environment = TaskEnvironment(_file_task())
    with environment:
        root = environment.root
        assert root is not None and root.is_dir()
        assert environment.initial_state["files"]["input.txt"]["content"] == "start"
        environment.resolve_path("input.txt", must_exist=True).write_text(
            "changed",
            encoding="utf-8",
        )
        assert environment.snapshot()["files"]["input.txt"]["content"] == "changed"
    assert root is not None and not root.exists()


def test_environment_rejects_path_escape():
    with TaskEnvironment(_file_task()) as environment:
        for path in ["../outside.txt", "C:/outside.txt", "/outside.txt"]:
            try:
                environment.resolve_path(path)
            except PermissionError:
                pass
            else:
                raise AssertionError(f"路径逃逸未被拒绝: {path}")


def test_registry_enforces_permissions():
    registry = create_default_skill_registry()
    with TaskEnvironment(
        _file_task(),
        allowed_permissions={"compute"},
    ) as environment:
        handler = registry.handlers_for(environment)["append_text_file"]
        try:
            handler(path="input.txt", content="more")
        except PermissionError as exc:
            assert "file:read" in str(exc)
            assert "file:write" in str(exc)
        else:
            raise AssertionError("缺少文件权限时 handler 不应执行")

    with TaskEnvironment(_file_task(), allowed_permissions=set()) as environment:
        assert environment.allowed_permissions == frozenset()


def test_calculator_rejects_code_execution_syntax():
    registry = create_default_skill_registry()
    task = Task("runtime-calc", "calculate")
    with TaskEnvironment(task) as environment:
        handler = registry.handlers_for(environment)["evaluate_decimal_expression"]
        try:
            handler(expression="__import__('os').getcwd()")
        except ValueError as exc:
            assert "不允许" in str(exc)
        else:
            raise AssertionError("表达式 handler 不应执行函数调用")


if __name__ == "__main__":
    test_environment_setup_snapshot_and_teardown()
    test_environment_rejects_path_escape()
    test_registry_enforces_permissions()
    test_calculator_rejects_code_execution_syntax()
    print("execution runtime tests passed")
