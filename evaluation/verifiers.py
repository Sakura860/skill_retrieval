"""确定性任务完成度验收协议与内置实现。"""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from core.schemas import Task, TaskEvaluationConfig

_MISSING = object()


@dataclass(frozen=True)
class VerificationContext:
    """一次验收所需的任务、状态、输出和执行证据。"""

    task: Task
    initial_state: Any = None
    final_state: Any = None
    agent_output: Any = None
    trajectory: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class VerifierResult:
    """可解释、可序列化的任务验收结果。"""

    passed: bool
    verifier_type: str
    failure_reason: str | None = None
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] = field(default_factory=dict)


class TaskVerifier(ABC):
    """确定性任务验收器接口。"""

    verifier_type: str

    @abstractmethod
    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        """根据最终输出或环境状态判断任务是否完成。"""

    def _passed(
        self,
        config: TaskEvaluationConfig,
        expected: Any,
        actual: Any,
        **details: Any,
    ) -> VerifierResult:
        effect_failure = _verify_effects(context_details=details, config=config)
        if effect_failure is not None:
            return VerifierResult(
                passed=False,
                verifier_type=self.verifier_type,
                failure_reason=effect_failure,
                expected=expected,
                actual=actual,
                details=details,
            )
        return VerifierResult(
            passed=True,
            verifier_type=self.verifier_type,
            expected=expected,
            actual=actual,
            details=details,
        )

    def _failed(
        self,
        reason: str,
        expected: Any = None,
        actual: Any = None,
        **details: Any,
    ) -> VerifierResult:
        return VerifierResult(
            passed=False,
            verifier_type=self.verifier_type,
            failure_reason=reason,
            expected=expected,
            actual=actual,
            details=details,
        )


class ExactMatchVerifier(TaskVerifier):
    """验收数字或字符串输出；数字可配置绝对误差。"""

    verifier_type = "exact_match"

    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        expected = config.expected_output
        if expected is None:
            return self._failed("missing_expected_output")
        actual = context.agent_output
        if not _exact_equal(actual, expected, config.tolerance):
            return self._failed("output_mismatch", expected, actual)
        effects = _collect_effects(context)
        return self._passed(config, expected, actual, effects=effects)


class StructuredJsonVerifier(TaskVerifier):
    """验收完整 JSON 结构，可同时检查输出和最终状态。"""

    verifier_type = "json_match"

    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        checks: list[tuple[str, Any, Any]] = []
        if config.expected_output is not None:
            try:
                actual_output = _as_json(context.agent_output)
                expected_output = _as_json(config.expected_output)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return self._failed("invalid_json_output", config.expected_output, context.agent_output, error=str(exc))
            checks.append(("output", expected_output, actual_output))
        if config.expected_state is not None:
            checks.append(("state", config.expected_state, context.final_state))
        if not checks:
            return self._failed("missing_expected_json")

        for target, expected, actual in checks:
            if not _structured_equal(actual, expected, config.tolerance):
                return self._failed(
                    f"{target}_mismatch",
                    expected,
                    actual,
                    target=target,
                )
        effects = _collect_effects(context)
        return self._passed(
            config,
            {target: expected for target, expected, _ in checks},
            {target: actual for target, _, actual in checks},
            effects=effects,
        )


class FileStateVerifier(TaskVerifier):
    """根据受控环境提供的文件快照验收内容与副作用。"""

    verifier_type = "file_state"

    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        if config.expected_state is None:
            return self._failed("missing_expected_state")
        expected_files = _file_map(config.expected_state)
        actual_files = _file_map(context.final_state)
        for path, expected in expected_files.items():
            actual = actual_files.get(path, _MISSING)
            matched, normalized_actual = _file_entry_matches(
                actual,
                expected,
                config.tolerance,
            )
            if not matched:
                return self._failed(
                    "file_state_mismatch",
                    {path: expected},
                    {path: None if actual is _MISSING else normalized_actual},
                    path=path,
                )
        effects = _changed_file_paths(context.initial_state, context.final_state)
        effect_failure = _effect_failure(config, effects)
        if effect_failure is not None:
            return self._failed(
                effect_failure,
                config.expected_state,
                context.final_state,
                effects=effects,
            )
        return VerifierResult(
            passed=True,
            verifier_type=self.verifier_type,
            expected=config.expected_state,
            actual=context.final_state,
            details={"effects": effects},
        )


class SQLiteStateVerifier(TaskVerifier):
    """以只读查询验收隔离 SQLite 数据库或已捕获的查询结果。"""

    verifier_type = "sqlite_state"

    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        spec = config.expected_state
        if not isinstance(spec, dict):
            return self._failed("missing_expected_state")

        expected_rows = spec.get("expected_rows", config.expected_output)
        if expected_rows is None:
            return self._failed("missing_expected_rows")

        final_state = context.final_state if isinstance(context.final_state, dict) else {}
        try:
            if "query_result" in final_state:
                actual_rows = final_state["query_result"]
            else:
                actual_rows = self._query_database(final_state, spec, expected_rows)
        except (OSError, sqlite3.Error, ValueError) as exc:
            return self._failed("sqlite_verification_error", expected_rows, None, error=str(exc))

        ordered = bool(spec.get("ordered", True))
        if not ordered:
            expected_rows = _sort_rows(expected_rows)
            actual_rows = _sort_rows(actual_rows)
        if not _structured_equal(actual_rows, expected_rows, config.tolerance):
            return self._failed("sqlite_result_mismatch", expected_rows, actual_rows)

        effects = _collect_effects(context)
        effect_failure = _effect_failure(config, effects)
        if effect_failure is not None:
            return self._failed(effect_failure, expected_rows, actual_rows, effects=effects)
        return VerifierResult(
            passed=True,
            verifier_type=self.verifier_type,
            expected=expected_rows,
            actual=actual_rows,
            details={"effects": effects},
        )

    @staticmethod
    def _query_database(
        final_state: dict,
        spec: dict,
        expected_rows: Any,
    ) -> list[Any]:
        database_path = final_state.get("database_path")
        query = spec.get("query")
        if not database_path or not query:
            raise ValueError("final_state.database_path 和 expected_state.query 均为必填")
        path = Path(database_path).resolve()
        if not path.is_file():
            raise ValueError(f"SQLite 数据库不存在: {path}")

        uri = f"file:{path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            cursor = connection.execute(query, tuple(spec.get("params", [])))
            rows = cursor.fetchall()
            expects_dict_rows = (
                isinstance(expected_rows, list)
                and bool(expected_rows)
                and isinstance(expected_rows[0], dict)
            )
            if rows and expects_dict_rows:
                columns = [item[0] for item in cursor.description or []]
                return [dict(zip(columns, row)) for row in rows]
            return [list(row) for row in rows]


class HTTPStateVerifier(TaskVerifier):
    """Verify loopback HTTP output plus the exact request/response trace."""

    verifier_type = "http_state"

    def verify(
        self,
        context: VerificationContext,
        config: TaskEvaluationConfig,
    ) -> VerifierResult:
        if not isinstance(config.expected_state, dict):
            return self._failed("missing_expected_state")
        final_state = context.final_state if isinstance(context.final_state, dict) else {}
        actual_http = final_state.get("http")
        if not isinstance(actual_http, dict):
            return self._failed("missing_http_state", config.expected_state, actual_http)
        if not _structured_equal(actual_http, config.expected_state, config.tolerance):
            return self._failed(
                "http_state_mismatch",
                config.expected_state,
                actual_http,
            )

        expected_output = config.expected_output
        actual_output = context.agent_output
        if expected_output is not None:
            if isinstance(expected_output, (dict, list)):
                try:
                    actual_output = _as_json(actual_output)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    return self._failed(
                        "invalid_json_output",
                        expected_output,
                        context.agent_output,
                        error=str(exc),
                    )
                output_matches = _structured_equal(
                    actual_output,
                    expected_output,
                    config.tolerance,
                )
            else:
                output_matches = _exact_equal(
                    actual_output,
                    expected_output,
                    config.tolerance,
                )
            if not output_matches:
                return self._failed(
                    "output_mismatch",
                    expected_output,
                    actual_output,
                )

        return VerifierResult(
            passed=True,
            verifier_type=self.verifier_type,
            expected={
                "output": expected_output,
                "http": config.expected_state,
            },
            actual={"output": actual_output, "http": actual_http},
            details={"request_count": len(actual_http.get("requests", []))},
        )


class TaskVerifierRegistry:
    """按 verifier_type 管理确定性验收器。"""

    def __init__(self, verifiers: list[TaskVerifier] | None = None):
        self._verifiers: dict[str, TaskVerifier] = {}
        for verifier in verifiers or []:
            self.register(verifier)

    def register(self, verifier: TaskVerifier) -> None:
        if not verifier.verifier_type:
            raise ValueError("verifier_type 不能为空")
        self._verifiers[verifier.verifier_type] = verifier

    def get(self, verifier_type: str) -> TaskVerifier | None:
        return self._verifiers.get(verifier_type)

    @classmethod
    def default(cls) -> "TaskVerifierRegistry":
        return cls([
            ExactMatchVerifier(),
            StructuredJsonVerifier(),
            FileStateVerifier(),
            SQLiteStateVerifier(),
            HTTPStateVerifier(),
        ])


DEFAULT_VERIFIER_REGISTRY = TaskVerifierRegistry.default()


def verify_task(
    task: Task,
    result: dict,
    registry: TaskVerifierRegistry | None = None,
) -> VerifierResult | None:
    """使用任务私有评测配置验收；无配置和 ground truth 时返回 None。"""
    config = task.evaluation
    verifier_type = config.verifier_type if config is not None else None
    if config is None and task.ground_truth is not None:
        config = TaskEvaluationConfig(
            verifier_type="exact_match",
            expected_output=task.ground_truth,
        )
        verifier_type = "ground_truth"
    if config is None:
        return None

    verifier = (registry or DEFAULT_VERIFIER_REGISTRY).get(config.verifier_type)
    if verifier is None:
        return VerifierResult(
            passed=False,
            verifier_type=config.verifier_type,
            failure_reason="unknown_verifier_type",
        )
    context = VerificationContext(
        task=task,
        initial_state=result.get("initial_state"),
        final_state=result.get("final_state"),
        agent_output=result.get("answer"),
        trajectory=list(result.get("trajectory", [])),
    )
    verification = verifier.verify(context, config)
    if verifier_type == "ground_truth":
        return VerifierResult(
            passed=verification.passed,
            verifier_type="ground_truth",
            failure_reason=verification.failure_reason,
            expected=verification.expected,
            actual=verification.actual,
            details=verification.details,
        )
    return verification


def _as_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    raise TypeError(f"不支持的 JSON 类型: {type(value).__name__}")


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None


def _exact_equal(actual: Any, expected: Any, tolerance: float | None) -> bool:
    actual_number = _decimal(actual)
    expected_number = _decimal(expected)
    if actual_number is not None and expected_number is not None:
        allowed = Decimal(str(tolerance or 0))
        return abs(actual_number - expected_number) <= allowed
    return str(actual).strip() == str(expected).strip()


def _structured_equal(actual: Any, expected: Any, tolerance: float | None) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _structured_equal(actual[key], value, tolerance)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _structured_equal(left, right, tolerance)
            for left, right in zip(actual, expected)
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return _exact_equal(actual, expected, tolerance)
    return type(actual) is type(expected) and actual == expected


def _file_map(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    files = state.get("files", state)
    return files if isinstance(files, dict) else {}


def _file_entry_matches(
    actual: Any,
    expected: Any,
    tolerance: float | None,
) -> tuple[bool, Any]:
    if isinstance(expected, dict) and any(
        key in expected for key in ("exists", "content", "json")
    ):
        expected_exists = bool(expected.get("exists", True))
        if actual is _MISSING:
            return (not expected_exists), None
        actual_entry = actual if isinstance(actual, dict) else {"exists": True, "content": actual}
        if bool(actual_entry.get("exists", True)) != expected_exists:
            return False, actual_entry
        if not expected_exists:
            return True, actual_entry
        if "content" in expected and not _exact_equal(
            actual_entry.get("content"),
            expected["content"],
            tolerance,
        ):
            return False, actual_entry
        if "json" in expected:
            try:
                actual_json = (
                    actual_entry["json"]
                    if "json" in actual_entry
                    else _as_json(actual_entry.get("content"))
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                return False, actual_entry
            if not _structured_equal(actual_json, expected["json"], tolerance):
                return False, actual_entry
        return True, actual_entry
    if actual is _MISSING:
        return False, None
    actual_value = actual.get("content") if isinstance(actual, dict) else actual
    return _structured_equal(actual_value, expected, tolerance), actual_value


def _changed_file_paths(initial_state: Any, final_state: Any) -> list[str]:
    initial = _file_map(initial_state)
    final = _file_map(final_state)
    return sorted(
        path
        for path in initial.keys() | final.keys()
        if initial.get(path, _MISSING) != final.get(path, _MISSING)
    )


def _collect_effects(context: VerificationContext) -> list[Any]:
    if isinstance(context.final_state, dict) and isinstance(
        context.final_state.get("effects"),
        list,
    ):
        return context.final_state["effects"]
    if isinstance(context.initial_state, dict) and isinstance(context.final_state, dict):
        return sorted(
            key
            for key in context.initial_state.keys() | context.final_state.keys()
            if context.initial_state.get(key, _MISSING)
            != context.final_state.get(key, _MISSING)
        )
    return []


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _effect_failure(
    config: TaskEvaluationConfig,
    actual_effects: list[Any],
) -> str | None:
    actual = {_canonical(effect) for effect in actual_effects}
    required = {_canonical(effect) for effect in config.required_effects}
    forbidden = {_canonical(effect) for effect in config.forbidden_effects}
    if not required.issubset(actual):
        return "missing_required_effect"
    if actual & forbidden:
        return "forbidden_effect_detected"
    return None


def _verify_effects(
    context_details: dict[str, Any],
    config: TaskEvaluationConfig,
) -> str | None:
    effects = context_details.get("effects", [])
    return _effect_failure(config, effects)


def _sort_rows(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    return sorted(rows, key=_canonical)
