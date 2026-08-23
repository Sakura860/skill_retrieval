"""benchmark_v01 使用的确定性、安全 Skill handler。"""
from __future__ import annotations

import ast
import csv
import json
import operator
import re
import shutil
from contextlib import closing
from decimal import Decimal, InvalidOperation
from io import StringIO
from typing import Any

from .environment import TaskEnvironment
from .registry import SkillRegistry

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_READ_QUERY = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_AGGREGATE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise ValueError(f"不是有效数值: {value}") from exc


def _number_output(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value.normalize())


def _evaluate_expression(expression: str, *, integer_only: bool) -> Decimal:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式语法无效") from exc

    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("表达式只能包含数字")
            if integer_only and not isinstance(node.value, int):
                raise ValueError("整数表达式不能包含小数")
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            if integer_only and isinstance(node.op, ast.Div):
                raise ValueError("整数表达式只允许 //，不允许 /")
            left = visit(node.left)
            right = visit(node.right)
            if right == 0 and isinstance(node.op, (ast.Div, ast.FloorDiv)):
                raise ZeroDivisionError("除数不能为 0")
            return binary[type(node.op)](left, right)
        raise ValueError("表达式包含不允许的语法")

    result = visit(tree)
    if integer_only and result != result.to_integral_value():
        raise ValueError("整数表达式结果不是整数")
    return result


def evaluate_integer_expression(
    environment: TaskEnvironment,
    expression: str,
) -> int:
    return int(_evaluate_expression(expression, integer_only=True))


def evaluate_decimal_expression(
    environment: TaskEnvironment,
    expression: str,
    precision: int | None = None,
) -> int | float:
    result = _evaluate_expression(expression, integer_only=False)
    if precision is not None:
        if precision < 0:
            raise ValueError("precision 不能小于 0")
        quantum = Decimal(1).scaleb(-precision)
        result = result.quantize(quantum)
    return _number_output(result)


def calculate_percentage_change(
    environment: TaskEnvironment,
    old_value: Any,
    new_value: Any,
) -> int | float:
    old = _decimal(old_value)
    if old == 0:
        raise ValueError("old_value 不能为 0")
    result = (_decimal(new_value) - old) / abs(old) * Decimal(100)
    return _number_output(result)


def solve_linear_equation(
    environment: TaskEnvironment,
    a: Any,
    b: Any,
    c: Any,
) -> int | float:
    coefficient = _decimal(a)
    if coefficient == 0:
        raise ValueError("a 不能为 0")
    return _number_output((_decimal(c) - _decimal(b)) / coefficient)


def convert_measurement_units(
    environment: TaskEnvironment,
    value: Any,
    from_unit: str,
    to_unit: str,
) -> int | float:
    factors = {
        "mm": Decimal("0.001"),
        "毫米": Decimal("0.001"),
        "cm": Decimal("0.01"),
        "厘米": Decimal("0.01"),
        "m": Decimal("1"),
        "米": Decimal("1"),
        "km": Decimal("1000"),
        "千米": Decimal("1000"),
        "公里": Decimal("1000"),
    }
    source = from_unit.strip().lower()
    target = to_unit.strip().lower()
    if source not in factors or target not in factors:
        raise ValueError("只支持毫米、厘米、米和千米之间换算")
    result = _decimal(value) * factors[source] / factors[target]
    return _number_output(result)


def aggregate_number_list(
    environment: TaskEnvironment,
    values: list[Any],
    operation: str,
) -> int | float:
    numbers = [_decimal(value) for value in values]
    if not numbers:
        raise ValueError("values 不能为空")
    if operation == "sum":
        result = sum(numbers, Decimal(0))
    elif operation == "mean":
        result = sum(numbers, Decimal(0)) / len(numbers)
    elif operation == "min":
        result = min(numbers)
    elif operation == "max":
        result = max(numbers)
    else:
        raise ValueError("operation 必须是 sum、mean、min 或 max")
    return _number_output(result)


def _json_value(value: Any, expected_type: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("输入不是有效 JSON") from exc
    if not isinstance(value, expected_type):
        raise TypeError(f"输入必须是 {expected_type.__name__}")
    return value


def _json_output(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_json_fields(
    environment: TaskEnvironment,
    data: Any,
    fields: list[str],
) -> str:
    obj = _json_value(data, dict)
    missing = [field for field in fields if field not in obj]
    if missing:
        raise KeyError(f"JSON 缺少字段: {', '.join(missing)}")
    return _json_output({field: obj[field] for field in fields})


def rename_json_keys(
    environment: TaskEnvironment,
    data: Any,
    mapping: dict[str, str],
) -> str:
    obj = _json_value(data, dict)
    if not isinstance(mapping, dict):
        raise TypeError("mapping 必须是对象")
    result: dict[str, Any] = {}
    for key, value in obj.items():
        target = mapping.get(key, key)
        if target in result:
            raise ValueError(f"重命名后键冲突: {target}")
        result[target] = value
    return _json_output(result)


def filter_json_records(
    environment: TaskEnvironment,
    records: Any,
    field: str,
    equals: Any,
) -> str:
    items = _json_value(records, list)
    if not all(isinstance(item, dict) for item in items):
        raise TypeError("records 每一项都必须是对象")
    return _json_output([item for item in items if item.get(field) == equals])


def merge_json_objects(
    environment: TaskEnvironment,
    left: Any,
    right: Any,
    conflict_policy: str,
) -> str:
    left_obj = _json_value(left, dict)
    right_obj = _json_value(right, dict)
    conflicts = left_obj.keys() & right_obj.keys()
    if conflicts and conflict_policy == "error":
        raise ValueError(f"键冲突: {', '.join(sorted(conflicts))}")
    if conflict_policy not in {"left", "right", "error"}:
        raise ValueError("conflict_policy 必须是 left、right 或 error")
    result = dict(right_obj)
    result.update(left_obj)
    if conflict_policy == "right":
        result.update(right_obj)
    return _json_output(result)


def sort_json_records(
    environment: TaskEnvironment,
    records: Any,
    field: str,
    descending: bool = False,
) -> str:
    items = _json_value(records, list)
    if not all(isinstance(item, dict) and field in item for item in items):
        raise KeyError(f"部分记录缺少排序字段: {field}")
    return _json_output(
        sorted(items, key=lambda item: item[field], reverse=bool(descending))
    )


def convert_csv_to_json(
    environment: TaskEnvironment,
    csv_text: str,
    delimiter: str = ",",
) -> str:
    if len(delimiter) != 1:
        raise ValueError("delimiter 必须是单个字符")
    reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("CSV 缺少表头")
    return _json_output(list(reader))


def create_text_file(
    environment: TaskEnvironment,
    path: str,
    content: str,
) -> str:
    target = environment.resolve_path(path)
    if target.exists():
        raise FileExistsError(f"目标文件已存在: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return path


def overwrite_text_file(
    environment: TaskEnvironment,
    path: str,
    content: str,
) -> str:
    target = environment.resolve_path(path, must_exist=True)
    if not target.is_file():
        raise ValueError("目标不是文件")
    target.write_text(str(content), encoding="utf-8")
    return path


def append_text_file(
    environment: TaskEnvironment,
    path: str,
    content: str,
) -> str:
    target = environment.resolve_path(path, must_exist=True)
    if not target.is_file():
        raise ValueError("目标不是文件")
    with target.open("a", encoding="utf-8") as stream:
        stream.write(str(content))
    return path


def copy_file_preserve_source(
    environment: TaskEnvironment,
    source: str,
    destination: str,
) -> str:
    source_path = environment.resolve_path(source, must_exist=True)
    destination_path = environment.resolve_path(destination)
    if destination_path.exists():
        raise FileExistsError(f"目标文件已存在: {destination}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return destination


def move_file(
    environment: TaskEnvironment,
    source: str,
    destination: str,
) -> str:
    source_path = environment.resolve_path(source, must_exist=True)
    destination_path = environment.resolve_path(destination)
    if destination_path.exists():
        raise FileExistsError(f"目标文件已存在: {destination}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(destination_path)
    return destination


def patch_json_file(
    environment: TaskEnvironment,
    path: str,
    patch: dict[str, Any],
) -> str:
    target = environment.resolve_path(path, must_exist=True)
    if not isinstance(patch, dict):
        raise TypeError("patch 必须是对象")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("目标文件不是有效 JSON") from exc
    if not isinstance(data, dict):
        raise TypeError("目标 JSON 必须是对象")
    data.update(patch)
    target.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return _json_output(data)


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"非法 SQLite 标识符: {value}")
    return value


def _read_query(
    environment: TaskEnvironment,
    query: str,
    parameters: list[Any] | tuple[Any, ...] | None,
) -> list[list[Any]]:
    if not _READ_QUERY.match(query) or ";" in query.rstrip().rstrip(";"):
        raise PermissionError("只允许单条只读 SQLite 查询")
    with closing(environment.connect_sqlite(read_only=True)) as connection:
        rows = connection.execute(query, tuple(parameters or [])).fetchall()
    return [list(row) for row in rows]


def select_sqlite_rows(
    environment: TaskEnvironment,
    query: str,
    parameters: list[Any] | None = None,
) -> str:
    return _json_output(_read_query(environment, query, parameters))


def insert_sqlite_row(
    environment: TaskEnvironment,
    table: str,
    values: dict[str, Any],
) -> str:
    if not values:
        raise ValueError("values 不能为空")
    table_name = _identifier(table)
    columns = [_identifier(column) for column in values]
    placeholders = ", ".join("?" for _ in columns)
    sql = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    with closing(environment.connect_sqlite()) as connection:
        cursor = connection.execute(sql, tuple(values[column] for column in columns))
        connection.commit()
        row_id = cursor.lastrowid
    return _json_output({"inserted": 1, "row_id": row_id})


def update_sqlite_rows(
    environment: TaskEnvironment,
    table: str,
    where: dict[str, Any],
    changes: dict[str, Any],
) -> str:
    if not where or not changes:
        raise ValueError("where 和 changes 不能为空")
    table_name = _identifier(table)
    change_columns = [_identifier(column) for column in changes]
    where_columns = [_identifier(column) for column in where]
    set_sql = ", ".join(f"{column}=?" for column in change_columns)
    where_sql = " AND ".join(f"{column}=?" for column in where_columns)
    params = [changes[column] for column in change_columns]
    params.extend(where[column] for column in where_columns)
    with closing(environment.connect_sqlite()) as connection:
        cursor = connection.execute(
            f"UPDATE {table_name} SET {set_sql} WHERE {where_sql}",
            tuple(params),
        )
        connection.commit()
        updated = cursor.rowcount
    return _json_output({"updated": updated})


def delete_sqlite_rows(
    environment: TaskEnvironment,
    table: str,
    where: dict[str, Any],
) -> str:
    if not where:
        raise ValueError("禁止无条件删除")
    table_name = _identifier(table)
    where_columns = [_identifier(column) for column in where]
    where_sql = " AND ".join(f"{column}=?" for column in where_columns)
    with closing(environment.connect_sqlite()) as connection:
        cursor = connection.execute(
            f"DELETE FROM {table_name} WHERE {where_sql}",
            tuple(where[column] for column in where_columns),
        )
        connection.commit()
        deleted = cursor.rowcount
    return _json_output({"deleted": deleted})


def aggregate_sqlite_query(
    environment: TaskEnvironment,
    query: str,
    parameters: list[Any] | None = None,
) -> str:
    if not _AGGREGATE.search(query):
        raise ValueError("查询必须包含聚合函数")
    return _json_output(_read_query(environment, query, parameters))


def join_sqlite_tables(
    environment: TaskEnvironment,
    query: str,
    parameters: list[Any] | None = None,
) -> str:
    if not re.search(r"\bJOIN\b", query, re.IGNORECASE):
        raise ValueError("查询必须包含 JOIN")
    return _json_output(_read_query(environment, query, parameters))


def create_default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for name, handler in {
        "evaluate_integer_expression": evaluate_integer_expression,
        "evaluate_decimal_expression": evaluate_decimal_expression,
        "calculate_percentage_change": calculate_percentage_change,
        "solve_linear_equation": solve_linear_equation,
        "convert_measurement_units": convert_measurement_units,
        "aggregate_number_list": aggregate_number_list,
        "calculator": evaluate_decimal_expression,
    }.items():
        registry.register(name, handler, {"compute"})

    for name, handler in {
        "extract_json_fields": extract_json_fields,
        "rename_json_keys": rename_json_keys,
        "filter_json_records": filter_json_records,
        "merge_json_objects": merge_json_objects,
        "sort_json_records": sort_json_records,
        "convert_csv_to_json": convert_csv_to_json,
    }.items():
        registry.register(name, handler, {"transform"})

    for name, handler, permissions in [
        ("create_text_file", create_text_file, {"file:write"}),
        ("overwrite_text_file", overwrite_text_file, {"file:read", "file:write"}),
        ("append_text_file", append_text_file, {"file:read", "file:write"}),
        ("copy_file_preserve_source", copy_file_preserve_source, {"file:read", "file:write"}),
        ("move_file", move_file, {"file:read", "file:write"}),
        ("patch_json_file", patch_json_file, {"file:read", "file:write"}),
    ]:
        registry.register(name, handler, permissions)

    for name, handler, permissions in [
        ("select_sqlite_rows", select_sqlite_rows, {"sqlite:read"}),
        ("insert_sqlite_row", insert_sqlite_row, {"sqlite:write"}),
        ("update_sqlite_rows", update_sqlite_rows, {"sqlite:write"}),
        ("delete_sqlite_rows", delete_sqlite_rows, {"sqlite:write"}),
        ("aggregate_sqlite_query", aggregate_sqlite_query, {"sqlite:read"}),
        ("join_sqlite_tables", join_sqlite_tables, {"sqlite:read"}),
    ]:
        registry.register(name, handler, permissions)
    return registry
