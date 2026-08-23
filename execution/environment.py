"""每个 Task 独占、可重复初始化和清理的执行环境。"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.schemas import Task

DEFAULT_PERMISSIONS = frozenset({
    "compute",
    "transform",
    "file:read",
    "file:write",
    "sqlite:read",
    "sqlite:write",
})


def load_environment_fixtures(path: str | Path | None) -> dict[str, dict]:
    """读取可信的环境 fixture 定义；未提供路径时返回空表。"""
    if path is None:
        return {}
    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"环境 fixture 文件不存在: {fixture_path}")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("环境 fixture 顶层必须是对象")
    return data


class TaskEnvironment:
    """把文件和 SQLite 副作用限制在单任务临时目录中。"""

    def __init__(
        self,
        task: Task,
        fixtures: dict[str, dict] | None = None,
        allowed_permissions: set[str] | frozenset[str] | None = None,
    ):
        self.task = task
        self.fixtures = fixtures or {}
        self.allowed_permissions = frozenset(
            DEFAULT_PERMISSIONS
            if allowed_permissions is None
            else allowed_permissions
        )
        self._temporary_directory: tempfile.TemporaryDirectory | None = None
        self.root: Path | None = None
        self.database_path: Path | None = None
        self.initial_state: dict[str, Any] | None = None

    def __enter__(self) -> "TaskEnvironment":
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"skill-agent-{self.task.id}-"
        )
        self.root = Path(self._temporary_directory.name).resolve()
        try:
            self._setup()
            self.initial_state = self.snapshot()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = None
        self.root = None
        self.database_path = None

    def has_permissions(self, permissions: frozenset[str]) -> bool:
        return permissions.issubset(self.allowed_permissions)

    def resolve_path(self, relative_path: str, *, must_exist: bool = False) -> Path:
        """解析受控相对路径并拒绝绝对路径和目录穿越。"""
        root = self._require_root()
        candidate_input = Path(relative_path)
        if candidate_input.is_absolute():
            raise PermissionError("只允许受控目录内的相对路径")
        candidate = (root / candidate_input).resolve()
        if candidate == root or root not in candidate.parents:
            raise PermissionError("路径超出 Task 受控目录")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"文件不存在: {relative_path}")
        return candidate

    def connect_sqlite(self, *, read_only: bool = False) -> sqlite3.Connection:
        if self.database_path is None or not self.database_path.is_file():
            raise RuntimeError("当前 Task 没有初始化 SQLite 数据库")
        if read_only:
            uri = f"file:{self.database_path.as_posix()}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        return sqlite3.connect(self.database_path)

    def snapshot(self) -> dict[str, Any]:
        """捕获 verifier 所需的文件和 SQLite 最终状态。"""
        root = self._require_root()
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if self.database_path is not None and path == self.database_path:
                continue
            relative = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8")
            entry: dict[str, Any] = {"exists": True, "content": content}
            if path.suffix.lower() == ".json":
                try:
                    entry["json"] = json.loads(content)
                except json.JSONDecodeError:
                    pass
            files[relative] = entry

        state: dict[str, Any] = {"files": files}
        if self.database_path is not None:
            state["database_path"] = str(self.database_path)
            query_result = self._verification_query_result()
            if query_result is not None:
                state["query_result"] = query_result
        return state

    def initial_state_copy(self) -> dict[str, Any] | None:
        return deepcopy(self.initial_state)

    def _setup(self) -> None:
        metadata = self.task.metadata or {}
        environment = metadata.get("environment_fixture") or {}
        if not isinstance(environment, dict):
            raise ValueError("environment_fixture 必须是对象")
        initial_state = environment.get("initial_state") or {}
        self._setup_files(initial_state)

        fixture_id = environment.get("sqlite_fixture_id")
        if fixture_id:
            fixture = self.fixtures.get(fixture_id)
            if fixture is None:
                raise KeyError(f"未找到 SQLite fixture: {fixture_id}")
            self._setup_sqlite(fixture)

    def _setup_files(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("initial_state 必须是对象")
        files = state.get("files", {})
        if not isinstance(files, dict):
            raise ValueError("initial_state.files 必须是对象")
        for relative, raw_entry in files.items():
            entry = raw_entry if isinstance(raw_entry, dict) else {"content": raw_entry}
            if entry.get("exists", True) is False:
                continue
            path = self.resolve_path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if "json" in entry:
                content = json.dumps(
                    entry["json"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            else:
                content = str(entry.get("content", ""))
            path.write_text(content, encoding="utf-8")

    def _setup_sqlite(self, fixture: dict[str, Any]) -> None:
        if fixture.get("type") != "sqlite":
            raise ValueError("仅支持 sqlite 类型的数据库 fixture")
        setup_sql = fixture.get("setup_sql")
        if not isinstance(setup_sql, list) or not setup_sql:
            raise ValueError("SQLite fixture.setup_sql 必须是非空列表")
        self.database_path = self.resolve_path("task.db")
        with closing(sqlite3.connect(self.database_path)) as connection:
            for statement in setup_sql:
                connection.execute(str(statement))
            connection.commit()

    def _verification_query_result(self) -> list[Any] | None:
        evaluation = self.task.evaluation
        if evaluation is None or evaluation.verifier_type != "sqlite_state":
            return None
        spec = evaluation.expected_state
        if not isinstance(spec, dict) or not spec.get("query"):
            return None
        expected_rows = spec.get("expected_rows")
        with closing(self.connect_sqlite(read_only=True)) as connection:
            cursor = connection.execute(
                spec["query"],
                tuple(spec.get("params", [])),
            )
            rows = cursor.fetchall()
            if (
                isinstance(expected_rows, list)
                and expected_rows
                and isinstance(expected_rows[0], dict)
            ):
                columns = [item[0] for item in cursor.description or []]
                return [dict(zip(columns, row)) for row in rows]
            return [list(row) for row in rows]

    def _require_root(self) -> Path:
        if self.root is None:
            raise RuntimeError("TaskEnvironment 尚未进入上下文或已经关闭")
        return self.root
