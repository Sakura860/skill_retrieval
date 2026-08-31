"""Planner 参数使用的轻量 JSON Schema 校验。"""
from __future__ import annotations

from typing import Any


def validate_arguments(
    arguments: Any,
    schema: dict,
    task_instruction: str = "",
    task_inputs: dict[str, Any] | None = None,
) -> list[str]:
    """校验项目 Skill 使用的 JSON Schema 子集并返回可修复错误。"""
    if not isinstance(arguments, dict):
        return ["arguments 必须是 JSON 对象"]
    resolved, reference_errors = resolve_static_references(
        arguments,
        task_instruction,
        task_inputs or {},
    )
    if reference_errors:
        return reference_errors
    if not schema:
        return []
    return _validate(resolved, schema, "arguments")


def resolve_static_references(
    value: Any,
    task_instruction: str,
    task_inputs: dict[str, Any],
    path: str = "arguments",
) -> tuple[Any, list[str]]:
    """解析执行前已知的 $task/$input 引用，保留动态 $last_output。"""
    if isinstance(value, dict):
        resolved = {}
        errors: list[str] = []
        for key, item in value.items():
            result, item_errors = resolve_static_references(
                item,
                task_instruction,
                task_inputs,
                f"{path}.{key}",
            )
            resolved[key] = result
            errors.extend(item_errors)
        return resolved, errors
    if isinstance(value, list):
        resolved_items = []
        errors = []
        for index, item in enumerate(value):
            result, item_errors = resolve_static_references(
                item,
                task_instruction,
                task_inputs,
                f"{path}[{index}]",
            )
            resolved_items.append(result)
            errors.extend(item_errors)
        return resolved_items, errors
    if value == "$task":
        return task_instruction, []
    if isinstance(value, str) and value.startswith("$input."):
        key = value[len("$input."):]
        if key not in task_inputs:
            return value, [f"{path} 引用了未知任务输入 {key}"]
        return task_inputs[key], []
    return value, []


def validate_last_output_references(
    arguments: Any,
    parameter_schema: dict,
    previous_returns: dict | None,
    path: str = "arguments",
) -> list[str]:
    """检查 $last_output 与前一步 returns Schema 的结构兼容性。"""
    if arguments == "$last_output":
        if previous_returns is None:
            return [f"{path} 在第一步引用了不存在的 $last_output"]
        return _schema_compatibility_errors(previous_returns, parameter_schema, path)
    if isinstance(arguments, dict):
        properties = parameter_schema.get("properties", {}) if isinstance(
            parameter_schema, dict
        ) else {}
        errors = []
        for key, value in arguments.items():
            errors.extend(validate_last_output_references(
                value,
                properties.get(key, {}),
                previous_returns,
                f"{path}.{key}",
            ))
        return errors
    if isinstance(arguments, list):
        item_schema = parameter_schema.get("items", {}) if isinstance(
            parameter_schema, dict
        ) else {}
        errors = []
        for index, value in enumerate(arguments):
            errors.extend(validate_last_output_references(
                value,
                item_schema,
                previous_returns,
                f"{path}[{index}]",
            ))
        return errors
    return []


def _schema_compatibility_errors(
    source_schema: dict,
    target_schema: dict,
    path: str,
) -> list[str]:
    source_type = source_schema.get("type") if isinstance(source_schema, dict) else None
    target_type = target_schema.get("type") if isinstance(target_schema, dict) else None
    target_types = target_type if isinstance(target_type, list) else [target_type]
    compatible = (
        source_type is None
        or target_type is None
        or source_type in target_types
        or (source_type == "integer" and "number" in target_types)
    )
    if not compatible:
        return [
            f"{path} 的 $last_output 类型不兼容："
            f"前一步返回 {source_type}，参数要求 {target_type}"
        ]
    if source_type == "array" and target_type == "array":
        source_items = source_schema.get("items")
        target_items = target_schema.get("items")
        if isinstance(source_items, dict) and isinstance(target_items, dict):
            return _schema_compatibility_errors(
                source_items,
                target_items,
                f"{path}[]",
            )
    return []


def _validate(value: Any, schema: dict, path: str) -> list[str]:
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected_type = schema.get("type")
    if not _matches_type(value, expected_type):
        return [f"{path} 应为 {expected_type}，实际为 {_type_name(value)}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} 必须是 {schema['enum']} 之一")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required if isinstance(required, list) else []:
            if key not in value:
                errors.append(f"{path} 缺少必填参数 {key}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, item in value.items():
                if key in properties:
                    errors.extend(_validate(item, properties[key], f"{path}.{key}"))
                elif schema.get("additionalProperties") is False:
                    errors.append(f"{path} 包含未声明参数 {key}")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} 不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} 不能大于 {schema['maximum']}")
    return errors


def _matches_type(value: Any, expected_type: Any) -> bool:
    if expected_type is None:
        return True
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if value == "$last_output":
        return True
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    check = checks.get(expected_type)
    return True if check is None else check(value)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__
