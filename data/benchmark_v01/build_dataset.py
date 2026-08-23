"""生成 Flat 与 Hierarchical 定向评测数据 v0.1。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import Skill  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402


def object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def skill(
    skill_id: str,
    name: str,
    family: str,
    brief: str,
    boundary: str,
    properties: dict,
    required: list[str],
    returns: dict,
    positive: str,
    negative: str,
    dependencies: list[str] | None = None,
) -> dict:
    return {
        "id": skill_id,
        "name": name,
        "brief_description": brief,
        "detailed_description": f"适用边界：{boundary}",
        "category": family,
        "parameters": object_schema(properties, required),
        "returns": returns,
        "tags": [family, "确定性", "benchmark_v01"],
        "examples": [f"正例：{positive}", f"反例：{negative}"],
        "dependencies": dependencies or [],
        "metadata": {"dataset": "benchmark_v01", "family": family},
    }


STR = {"type": "string"}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}
ARR = {"type": "array"}
OBJ = {"type": "object"}


SKILLS = [
    skill(
        "scalc_int", "evaluate_integer_expression", "calculation",
        "计算一个数值表达式并返回结果。",
        "只接受整数、括号及 + - * // 运算；不接受小数、百分比、方程或单位换算。",
        {"expression": STR}, ["expression"], NUM,
        "计算 (18 + 7) * 4。", "计算 12.5 / 4，应使用小数表达式技能。",
    ),
    skill(
        "scalc_dec", "evaluate_decimal_expression", "calculation",
        "计算一个数值表达式并返回结果。",
        "接受含小数的 + - * / 表达式并保留精度；不处理百分比语义、方程或单位。",
        {"expression": STR, "precision": {"type": "integer", "minimum": 0}},
        ["expression"], NUM,
        "计算 12.5 / 4。", "求 80 增长到 100 的百分比，应使用百分比技能。",
    ),
    skill(
        "scalc_pct", "calculate_percentage_change", "calculation",
        "计算两个数值之间的变化结果。",
        "仅计算从 old_value 到 new_value 的百分比变化；不是普通除法或百分数取值。",
        {"old_value": NUM, "new_value": NUM}, ["old_value", "new_value"], NUM,
        "计算 80 增长到 100 的百分比变化。", "计算 80 / 100，应使用表达式技能。",
    ),
    skill(
        "scalc_eq", "solve_linear_equation", "calculation",
        "计算含未知量的数值关系并返回结果。",
        "只求一元一次方程 ax+b=c 的 x；不计算已知数表达式或统计列表。",
        {"a": NUM, "b": NUM, "c": NUM}, ["a", "b", "c"], NUM,
        "求 3x+6=21。", "计算 3*5+6，应使用表达式技能。",
    ),
    skill(
        "scalc_unit", "convert_measurement_units", "calculation",
        "计算一个数值并转换为目标形式。",
        "仅在给定长度单位之间换算；不进行无单位算术、百分比或货币换算。",
        {"value": NUM, "from_unit": STR, "to_unit": STR},
        ["value", "from_unit", "to_unit"], NUM,
        "把 2.5 千米换算成米。", "计算 2.5*1000 的裸表达式，应使用表达式技能。",
    ),
    skill(
        "scalc_agg", "aggregate_number_list", "calculation",
        "计算一组数值并返回汇总结果。",
        "只对显式数值列表执行 sum、mean、min、max；不解析表达式或解方程。",
        {"values": {"type": "array", "items": NUM}, "operation": {"type": "string", "enum": ["sum", "mean", "min", "max"]}},
        ["values", "operation"], NUM,
        "求 [5, 7, 11] 的总和。", "计算 (5+7)*11，应使用表达式技能。",
    ),
    skill(
        "sjson_extract", "extract_json_fields", "json_text",
        "转换 JSON 数据并返回新的 JSON。",
        "从单个 JSON 对象保留指定字段；不改字段名、不筛选数组记录。",
        {"data": OBJ, "fields": {"type": "array", "items": STR}},
        ["data", "fields"], OBJ,
        "从用户对象提取 name 和 age。", "把 user_name 改成 name，应使用重命名技能。",
    ),
    skill(
        "sjson_rename", "rename_json_keys", "json_text",
        "转换 JSON 数据并返回新的 JSON。",
        "按 mapping 重命名单个对象的键并保留未映射字段；不删除字段或筛选记录。",
        {"data": OBJ, "mapping": OBJ}, ["data", "mapping"], OBJ,
        "把 user_name 重命名为 name。", "只保留 name 和 age，应使用字段提取技能。",
    ),
    skill(
        "sjson_filter", "filter_json_records", "json_text",
        "转换 JSON 记录并返回新的 JSON。",
        "按一个字段的等值条件筛选对象数组；不排序、不合并对象。",
        {"records": ARR, "field": STR, "equals": {}},
        ["records", "field", "equals"], ARR,
        "保留 active=true 的记录。", "按 age 升序排列，应使用排序技能。",
    ),
    skill(
        "sjson_merge", "merge_json_objects", "json_text",
        "转换多个 JSON 数据并返回新的 JSON。",
        "合并两个对象；冲突键按 conflict_policy 处理；不连接或筛选对象数组。",
        {"left": OBJ, "right": OBJ, "conflict_policy": {"type": "string", "enum": ["left", "right", "error"]}},
        ["left", "right", "conflict_policy"], OBJ,
        "以右侧优先合并两个配置对象。", "合并两个记录数组，应使用其他技能。",
    ),
    skill(
        "sjson_sort", "sort_json_records", "json_text",
        "转换 JSON 记录并返回新的 JSON。",
        "按指定字段稳定排序对象数组；不筛选记录、不重命名字段。",
        {"records": ARR, "field": STR, "descending": BOOL},
        ["records", "field"], ARR,
        "按 score 降序排列记录。", "只保留 score=10 的记录，应使用筛选技能。",
    ),
    skill(
        "sjson_csv", "convert_csv_to_json", "json_text",
        "转换结构化文本并返回 JSON。",
        "把带表头的 CSV 文本解析为对象数组；不处理已经是 JSON 的输入。",
        {"csv_text": STR, "delimiter": STR}, ["csv_text"], ARR,
        "把 name,age CSV 转成 JSON 数组。", "排序 JSON 数组，应使用排序技能。",
    ),
    skill(
        "sfile_create", "create_text_file", "file_operation",
        "在受控目录中写入一个文件。",
        "仅当目标不存在时创建 UTF-8 文本文件；目标已存在必须失败，不覆盖也不追加。",
        {"path": STR, "content": STR}, ["path", "content"], STR,
        "新建不存在的 report.txt。", "替换已有 report.txt，应使用覆盖技能。",
    ),
    skill(
        "sfile_overwrite", "overwrite_text_file", "file_operation",
        "在受控目录中写入一个文件。",
        "仅覆盖已有 UTF-8 文本文件的全部内容；不创建新文件，也不追加。",
        {"path": STR, "content": STR}, ["path", "content"], STR,
        "把已有 report.txt 全部替换。", "给 report.txt 末尾增加一行，应使用追加技能。",
    ),
    skill(
        "sfile_append", "append_text_file", "file_operation",
        "在受控目录中写入一个文件。",
        "只在已有文本文件末尾追加内容；不替换原内容，不负责创建目标。",
        {"path": STR, "content": STR}, ["path", "content"], STR,
        "在日志末尾追加一行。", "重写整个日志，应使用覆盖技能。",
    ),
    skill(
        "sfile_copy", "copy_file_preserve_source", "file_operation",
        "在受控目录中复制一个文件。",
        "复制文件到不存在的目标并保留源文件；不移动源文件，不覆盖已有目标。",
        {"source": STR, "destination": STR}, ["source", "destination"], STR,
        "把 source.txt 复制为 backup.txt。", "复制后删除源文件，应使用移动技能。",
    ),
    skill(
        "sfile_move", "move_file", "file_operation",
        "在受控目录中转移一个文件。",
        "把源文件移动到不存在的目标，成功后源路径必须消失；不是复制操作。",
        {"source": STR, "destination": STR}, ["source", "destination"], STR,
        "把 draft.txt 移动为 final.txt。", "保留 draft.txt 的备份，应使用复制技能。",
    ),
    skill(
        "sfile_patch_json", "patch_json_file", "file_operation",
        "在受控目录中修改一个文件。",
        "读取已有 JSON 对象并只更新 patch 中的键；必须保留其他键，不处理纯文本。",
        {"path": STR, "patch": OBJ}, ["path", "patch"], OBJ,
        "只把 settings.json 的 theme 改为 dark。", "替换整个文本文件，应使用覆盖技能。",
    ),
    skill(
        "sdb_select", "select_sqlite_rows", "sqlite",
        "在 SQLite 中处理表记录并返回结果。",
        "执行参数化只读 SELECT，返回原始行；不写数据库、不做聚合专用计算。",
        {"query": STR, "parameters": ARR}, ["query"], ARR,
        "查询年龄不小于 18 的用户。", "新增用户，应使用插入技能。",
    ),
    skill(
        "sdb_insert", "insert_sqlite_row", "sqlite",
        "在 SQLite 中处理一条表记录。",
        "向指定表插入一行并拒绝覆盖现有主键；不更新或删除已有记录。",
        {"table": STR, "values": OBJ}, ["table", "values"], OBJ,
        "向 users 插入 Cara。", "修改 Bob 的状态，应使用更新技能。",
    ),
    skill(
        "sdb_update", "update_sqlite_rows", "sqlite",
        "在 SQLite 中处理已有表记录。",
        "按等值条件更新已有行的指定列；不插入缺失行，不删除记录。",
        {"table": STR, "where": OBJ, "changes": OBJ},
        ["table", "where", "changes"], OBJ,
        "把 Bob 的 active 更新为 1。", "删除 inactive 用户，应使用删除技能。",
    ),
    skill(
        "sdb_delete", "delete_sqlite_rows", "sqlite",
        "在 SQLite 中处理已有表记录。",
        "按等值条件删除行；不清空整表，不更新或插入记录。",
        {"table": STR, "where": OBJ}, ["table", "where"], OBJ,
        "删除 active=0 的用户。", "把 active 改为 1，应使用更新技能。",
    ),
    skill(
        "sdb_aggregate", "aggregate_sqlite_query", "sqlite",
        "在 SQLite 中查询表记录并返回结果。",
        "只执行 COUNT、SUM、AVG、MIN、MAX 聚合查询；不返回逐行明细，不写数据库。",
        {"query": STR, "parameters": ARR}, ["query"], ARR,
        "统计成年用户数量。", "列出成年用户姓名，应使用普通查询技能。",
    ),
    skill(
        "sdb_join", "join_sqlite_tables", "sqlite",
        "在 SQLite 中查询多张表并返回结果。",
        "执行涉及至少两张表的只读 JOIN；单表查询或写操作不适用。",
        {"query": STR, "parameters": ARR}, ["query"], ARR,
        "连接 users 与 orders 返回用户名和金额。", "只查询 users，应使用普通查询技能。",
    ),
]


BM25 = BM25Retriever(text_level="brief")
BM25.index([Skill(**item) for item in SKILLS])


def exact(expected: Any, tolerance: float | None = None) -> dict:
    config = {"verifier_type": "exact_match", "expected_output": expected}
    if tolerance is not None:
        config["tolerance"] = tolerance
    return config


def json_match(expected: Any) -> dict:
    return {"verifier_type": "json_match", "expected_output": expected}


def file_state(files: dict, required: list[str], forbidden: list[str]) -> dict:
    return {
        "verifier_type": "file_state",
        "expected_state": {"files": files},
        "required_effects": required,
        "forbidden_effects": forbidden,
    }


def sqlite_state(query: str, rows: list, params: list | None = None) -> dict:
    return {
        "verifier_type": "sqlite_state",
        "expected_state": {
            "query": query,
            "params": params or [],
            "expected_rows": rows,
            "ordered": True,
        },
    }


TASK_SPECS = [
    ("tcalc01", "dev", "calculation", "dev_integer_expression", "计算整数表达式 (18 + 7) * 4。", ["scalc_int"], exact(100), {}),
    ("tcalc02", "dev", "calculation", "dev_decimal_expression", "精确计算小数表达式 12.5 / 4。", ["scalc_dec"], exact(3.125), {}),
    ("tcalc03", "test", "calculation", "test_percentage_change", "一项指标从 80 增长到 100，计算百分比变化。", ["scalc_pct"], exact(25), {}),
    ("tcalc04", "dev", "calculation", "dev_linear_equation", "求一元一次方程 3x + 6 = 21 中的 x。", ["scalc_eq"], exact(5), {}),
    ("tcalc05", "test", "calculation", "test_unit_conversion", "把 2.5 千米换算成米。", ["scalc_unit"], exact(2500), {}),
    ("tcalc06", "dev", "calculation", "dev_number_aggregate", "求数值列表 [5, 7, 11, 19] 的总和。", ["scalc_agg"], exact(42), {}),
    ("tcalc07", "dev", "calculation", "dev_decimal_then_percentage", "先计算 12.5 + 7.5，再计算所得结果相对于 16 的增长百分比。", ["scalc_dec", "scalc_pct"], exact(25), {}),
    ("tcalc08", "test", "calculation", "test_unit_then_integer", "先把 3 千米换算成米，再给结果加上 250。", ["scalc_unit", "scalc_int"], exact(3250), {}),
    ("tjson01", "dev", "json_text", "dev_extract_fields", "从 {\"name\":\"Ada\",\"age\":37,\"city\":\"London\"} 中只提取 name 和 age。", ["sjson_extract"], json_match({"name": "Ada", "age": 37}), {}),
    ("tjson02", "test", "json_text", "test_rename_key", "把 {\"user_name\":\"Ada\",\"age\":37} 的 user_name 键重命名为 name，保留其他字段。", ["sjson_rename"], json_match({"name": "Ada", "age": 37}), {}),
    ("tjson03", "dev", "json_text", "dev_filter_records", "从用户数组 [{\"name\":\"Ada\",\"active\":true},{\"name\":\"Bob\",\"active\":false}] 中保留 active=true 的记录。", ["sjson_filter"], json_match([{"name": "Ada", "active": True}]), {}),
    ("tjson04", "dev", "json_text", "dev_merge_objects", "合并 {\"theme\":\"light\",\"lang\":\"zh\"} 与 {\"theme\":\"dark\"}，冲突时右侧优先。", ["sjson_merge"], json_match({"theme": "dark", "lang": "zh"}), {}),
    ("tjson05", "test", "json_text", "test_sort_records", "把 [{\"name\":\"A\",\"score\":8},{\"name\":\"B\",\"score\":10}] 按 score 降序排列。", ["sjson_sort"], json_match([{"name": "B", "score": 10}, {"name": "A", "score": 8}]), {}),
    ("tjson06", "dev", "json_text", "dev_csv_conversion", "把 CSV 文本 name,age\nAda,37\nBob,16 转为 JSON 对象数组。", ["sjson_csv"], json_match([{"name": "Ada", "age": "37"}, {"name": "Bob", "age": "16"}]), {}),
    ("tjson07", "dev", "json_text", "dev_extract_then_rename", "从 {\"user_name\":\"Ada\",\"age\":37,\"city\":\"London\"} 提取 user_name 和 age，再把 user_name 重命名为 name。", ["sjson_extract", "sjson_rename"], json_match({"name": "Ada", "age": 37}), {}),
    ("tjson08", "test", "json_text", "test_filter_then_sort", "先从 [{\"name\":\"A\",\"active\":true,\"score\":8},{\"name\":\"B\",\"active\":false,\"score\":10},{\"name\":\"C\",\"active\":true,\"score\":9}] 筛出 active=true，再按 score 降序排列。", ["sjson_filter", "sjson_sort"], json_match([{"name": "C", "active": True, "score": 9}, {"name": "A", "active": True, "score": 8}]), {}),
    ("tfile01", "dev", "file_operation", "dev_create_file", "在受控目录中新建 report.txt，内容为 hello。", ["sfile_create"], file_state({"report.txt": {"exists": True, "content": "hello"}}, ["report.txt"], []), {"initial_state": {"files": {}}}),
    ("tfile02", "test", "file_operation", "test_overwrite_file", "将已有 report.txt 的内容完整替换为 final。", ["sfile_overwrite"], file_state({"report.txt": {"exists": True, "content": "final"}}, ["report.txt"], []), {"initial_state": {"files": {"report.txt": {"exists": True, "content": "draft"}}}}),
    ("tfile03", "dev", "file_operation", "dev_append_file", "在已有 log.txt 末尾追加一行 done，保留原内容。", ["sfile_append"], file_state({"log.txt": {"exists": True, "content": "start\ndone"}}, ["log.txt"], []), {"initial_state": {"files": {"log.txt": {"exists": True, "content": "start"}}}}),
    ("tfile04", "dev", "file_operation", "dev_copy_file", "把 source.txt 复制为 backup.txt，并确保 source.txt 保持不变。", ["sfile_copy"], file_state({"source.txt": {"exists": True, "content": "data"}, "backup.txt": {"exists": True, "content": "data"}}, ["backup.txt"], ["source.txt"]), {"initial_state": {"files": {"source.txt": {"exists": True, "content": "data"}}}}),
    ("tfile05", "test", "file_operation", "test_move_file", "把 draft.txt 移动为 final.txt，完成后 draft.txt 不应存在。", ["sfile_move"], file_state({"draft.txt": {"exists": False}, "final.txt": {"exists": True, "content": "ready"}}, ["draft.txt", "final.txt"], []), {"initial_state": {"files": {"draft.txt": {"exists": True, "content": "ready"}}}}),
    ("tfile06", "dev", "file_operation", "dev_patch_json_file", "只把 settings.json 的 theme 改成 dark，同时保留 lang=zh。", ["sfile_patch_json"], file_state({"settings.json": {"exists": True, "json": {"theme": "dark", "lang": "zh"}}}, ["settings.json"], []), {"initial_state": {"files": {"settings.json": {"exists": True, "json": {"theme": "light", "lang": "zh"}}}}}),
    ("tfile07", "test", "file_operation", "test_copy_then_append", "先把 source.txt 复制为 backup.txt，再在 backup.txt 末尾追加一行 extra，不能修改 source.txt。", ["sfile_copy", "sfile_append"], file_state({"source.txt": {"exists": True, "content": "base"}, "backup.txt": {"exists": True, "content": "base\nextra"}}, ["backup.txt"], ["source.txt"]), {"initial_state": {"files": {"source.txt": {"exists": True, "content": "base"}}}}),
    ("tdb01", "dev", "sqlite", "dev_select_rows", "查询 users 表中 age>=18 的用户姓名，按姓名升序返回。", ["sdb_select"], sqlite_state("SELECT name FROM users WHERE age >= 18 ORDER BY name", [["Ada"], ["Cara"]]), {"sqlite_fixture_id": "users_basic"}),
    ("tdb02", "test", "sqlite", "test_insert_row", "向 users 表插入 id=4、name=Dora、age=22、active=1 的用户。", ["sdb_insert"], sqlite_state("SELECT id, name, age, active FROM users WHERE id=4", [[4, "Dora", 22, 1]]), {"sqlite_fixture_id": "users_basic"}),
    ("tdb03", "dev", "sqlite", "dev_update_rows", "把 users 表中 name=Bob 的 active 更新为 1。", ["sdb_update"], sqlite_state("SELECT active FROM users WHERE name='Bob'", [[1]]), {"sqlite_fixture_id": "users_basic"}),
    ("tdb04", "test", "sqlite", "test_delete_rows", "删除 users 表中 active=0 的记录。", ["sdb_delete"], sqlite_state("SELECT name FROM users ORDER BY name", [["Ada"], ["Cara"]]), {"sqlite_fixture_id": "users_basic"}),
    ("tdb05", "dev", "sqlite", "dev_aggregate_query", "统计 users 表中 age>=18 的用户数量。", ["sdb_aggregate"], sqlite_state("SELECT COUNT(*) FROM users WHERE age >= 18", [[2]]), {"sqlite_fixture_id": "users_basic"}),
    ("tdb06", "dev", "sqlite", "dev_join_query", "连接 users 和 orders 表，返回订单 id、用户名和金额，按订单 id 排序。", ["sdb_join"], sqlite_state("SELECT o.id, u.name, o.amount FROM orders o JOIN users u ON u.id=o.user_id ORDER BY o.id", [[101, "Ada", 50.0], [102, "Cara", 75.5]]), {"sqlite_fixture_id": "users_orders"}),
    ("tdb07", "test", "sqlite", "test_insert_update_select", "先向 users 插入 id=4、name=Dora、age=17、active=0，再把 Dora 的 active 更新为 1，最后查询 Dora 的完整记录。", ["sdb_insert", "sdb_update", "sdb_select"], sqlite_state("SELECT id, name, age, active FROM users WHERE name='Dora'", [[4, "Dora", 17, 1]]), {"sqlite_fixture_id": "users_basic"}),
    ("tcalc09", "test", "calculation", "test_percentage_change", "某数值由 50 变为 65，请给出它的百分比变化。", ["scalc_pct"], exact(30), {}),
    ("tjson09", "test", "json_text", "test_rename_key", "将 {\"account_name\":\"Lin\",\"age\":29} 中的 account_name 改名为 name，其余数据不要丢失。", ["sjson_rename"], json_match({"name": "Lin", "age": 29}), {}),
    ("tfile08", "test", "file_operation", "test_overwrite_file", "已有 status.txt，请用 approved 完整替换它原来的内容。", ["sfile_overwrite"], file_state({"status.txt": {"exists": True, "content": "approved"}}, ["status.txt"], []), {"initial_state": {"files": {"status.txt": {"exists": True, "content": "pending"}}}}),
    ("tdb08", "test", "sqlite", "test_insert_row", "在 users 表新增 id=5、name=Evan、age=31、active=1 的记录。", ["sdb_insert"], sqlite_state("SELECT id, name, age, active FROM users WHERE id=5", [[5, "Evan", 31, 1]]), {"sqlite_fixture_id": "users_basic"}),
]


SLICE_GRID = [
    (5, 1, 800), (10, 1, 800), (20, 1, 800),
    (5, 3, 800), (10, 3, 800), (20, 3, 800),
    (5, 5, 800), (10, 5, 800), (20, 5, 800),
    (5, 1, 1200), (10, 1, 1200), (20, 1, 1200),
    (5, 3, 1200), (10, 3, 1200), (20, 3, 1200),
    (5, 5, 1200), (10, 5, 1200), (20, 5, 1200),
    (5, 1, 2000), (10, 1, 2000), (20, 1, 2000),
    (5, 3, 2000), (10, 3, 2000), (20, 3, 2000),
    (5, 5, 2000), (10, 5, 2000), (20, 5, 2000),
    (5, 1, 1200), (10, 3, 1200), (20, 5, 1200),
    (5, 3, 800), (10, 5, 1200), (20, 1, 2000), (5, 5, 2000),
]


def candidate_fixture(
    instruction: str,
    expected_sequence: list[str],
    candidate_count: int,
    target_rank: int,
) -> tuple[list[str], int]:
    primary = expected_sequence[0]
    other_gold = expected_sequence[1:]
    raw_ranking = BM25.retrieve(instruction, top_k=len(SKILLS)).ranked_ids()
    distractors = [item for item in raw_ranking if item not in expected_sequence]
    candidates = (other_gold + distractors)[: candidate_count - 1]
    candidates.insert(target_rank - 1, primary)
    return candidates, raw_ranking.index(primary) + 1


TASKS = []
template_counts: dict[str, int] = {}
for index, spec in enumerate(TASK_SPECS):
    (
        task_id,
        split,
        family,
        template_id,
        instruction,
        expected_sequence,
        evaluation,
        environment_fixture,
    ) = spec
    candidate_count, target_rank, budget = SLICE_GRID[index]
    candidates, raw_bm25_rank = candidate_fixture(
        instruction,
        expected_sequence,
        candidate_count,
        target_rank,
    )
    template_counts[template_id] = template_counts.get(template_id, 0) + 1
    TASKS.append({
        "id": task_id,
        "instruction": instruction,
        "expected_skills": list(dict.fromkeys(expected_sequence)),
        "expected_skill_sequence": expected_sequence,
        "ground_truth": None,
        "metadata": {
            "dataset": "benchmark_v01",
            "split": split,
            "family": family,
            "template_id": template_id,
            "variant_id": template_counts[template_id],
            "step_count": len(expected_sequence),
            "contrast": "flat_vs_hierarchical_progressive_disclosure",
            "slice": {
                "candidate_count": candidate_count,
                "primary_gold_skill_id": expected_sequence[0],
                "target_gold_rank": target_rank,
                "context_budget_tokens": budget,
                "raw_bm25_primary_rank": raw_bm25_rank,
            },
            "candidate_skill_ids": candidates,
            "candidate_generation": "bm25_brief_then_gold_rank_control",
            "environment_fixture": environment_fixture,
        },
        "evaluation": evaluation,
    })


GRAPH_SKILLS = [
    skill("g_fetch", "fetch_dataset", "graph", "准备处理数据。", "下载原始数据，是解析步骤的直接前置。", {"source": STR}, ["source"], STR, "获取数据。", "解析本地数据。"),
    skill("g_parse", "parse_dataset", "graph", "准备处理数据。", "解析已获取的原始数据，必须在 fetch_dataset 后执行。", {"raw": STR}, ["raw"], ARR, "解析获取结果。", "直接发布报告。", ["g_fetch"]),
    skill("g_schema", "load_validation_schema", "graph", "准备处理数据。", "加载验证规则，是 validate_dataset 的一个前置分支。", {"name": STR}, ["name"], OBJ, "加载规则。", "直接验证数据。"),
    skill("g_validate", "validate_dataset", "graph", "检查处理数据。", "同时依赖已解析数据和验证规则。", {"records": ARR, "schema": OBJ}, ["records", "schema"], ARR, "验证解析结果。", "在解析前验证。", ["g_parse", "g_schema"]),
    skill("g_normalize", "normalize_dataset", "graph", "转换处理数据。", "只处理已验证记录，必须在 validate_dataset 后执行。", {"records": ARR}, ["records"], ARR, "标准化有效记录。", "跳过验证直接标准化。", ["g_validate"]),
    skill("g_summarize", "summarize_dataset", "graph", "汇总处理数据。", "只汇总已标准化记录。", {"records": ARR}, ["records"], OBJ, "汇总标准化记录。", "汇总原始文本。", ["g_normalize"]),
    skill("g_write", "write_dataset_report", "graph", "输出处理结果。", "把汇总结果写成报告，依赖 summarize_dataset。", {"summary": OBJ, "path": STR}, ["summary", "path"], STR, "写入汇总报告。", "直接写原始数据。", ["g_summarize"]),
    skill("g_approve", "approve_dataset_report", "graph", "确认处理结果。", "批准已写出的报告，是发布前置。", {"path": STR}, ["path"], BOOL, "批准报告。", "批准尚未生成的报告。", ["g_write"]),
    skill("g_publish", "publish_dataset_report", "graph", "输出处理结果。", "发布已经批准的报告。", {"path": STR}, ["path"], STR, "发布批准报告。", "绕过批准发布。", ["g_approve"]),
    skill("g_missing", "transform_with_external_dictionary", "graph", "转换处理数据。", "依赖当前 Skill 库中不存在的外部词典加载技能。", {"records": ARR}, ["records"], ARR, "用外部词典转换。", "无词典转换。", ["g_external_dictionary"]),
    skill("g_cycle_a", "cycle_stage_a", "graph", "执行循环诊断阶段。", "错误地依赖 cycle_stage_b，用于检测环。", {}, [], STR, "环诊断。", "正式数据处理。", ["g_cycle_b"]),
    skill("g_cycle_b", "cycle_stage_b", "graph", "执行循环诊断阶段。", "错误地依赖 cycle_stage_a，用于检测环。", {}, [], STR, "环诊断。", "正式数据处理。", ["g_cycle_a"]),
]


def graph_task(
    task_id: str,
    instruction: str,
    sequence: list[str],
    diagnostic: str,
    expected_issue: str | None = None,
) -> dict:
    normal_candidates = [
        "g_fetch", "g_parse", "g_schema", "g_validate", "g_normalize",
        "g_summarize", "g_write", "g_approve", "g_publish",
    ]
    if diagnostic == "missing_dependency":
        candidates = ["g_missing"]
    elif diagnostic == "cycle_detection":
        candidates = ["g_cycle_a", "g_cycle_b"]
    else:
        candidates = normal_candidates
    return {
        "id": task_id,
        "instruction": instruction,
        "expected_skills": list(dict.fromkeys(sequence)),
        "expected_skill_sequence": sequence,
        "ground_truth": None,
        "metadata": {
            "dataset": "graph_diagnostics_v01",
            "split": "diagnostic",
            "diagnostic_type": diagnostic,
            "expected_graph_issue": expected_issue,
            "candidate_skill_ids": candidates,
        },
        "evaluation": exact("ok"),
    }


GRAPH_TASKS = [
    graph_task("tg01", "获取并解析数据。", ["g_fetch", "g_parse"], "linear_dependency"),
    graph_task("tg02", "获取数据、解析，并加载规则完成验证。", ["g_fetch", "g_schema", "g_parse", "g_validate"], "branch_merge_dependency"),
    graph_task("tg03", "把数据处理到标准化完成。", ["g_fetch", "g_schema", "g_parse", "g_validate", "g_normalize"], "long_chain"),
    graph_task("tg04", "生成标准化数据的汇总。", ["g_fetch", "g_schema", "g_parse", "g_validate", "g_normalize", "g_summarize"], "long_chain"),
    graph_task("tg05", "生成并写出数据报告。", ["g_fetch", "g_schema", "g_parse", "g_validate", "g_normalize", "g_summarize", "g_write"], "long_chain"),
    graph_task("tg06", "写出并批准数据报告。", ["g_fetch", "g_schema", "g_parse", "g_validate", "g_normalize", "g_summarize", "g_write", "g_approve"], "strict_order"),
    graph_task("tg07", "完成数据报告并发布。", ["g_fetch", "g_schema", "g_parse", "g_validate", "g_normalize", "g_summarize", "g_write", "g_approve", "g_publish"], "strict_order"),
    graph_task("tg08", "使用外部词典转换数据。", ["g_missing"], "missing_dependency", "g_external_dictionary"),
    graph_task("tg09", "执行循环阶段 A 和 B。", ["g_cycle_a", "g_cycle_b"], "cycle_detection", "cycle"),
    graph_task("tg10", "验证数据但不要继续标准化。", ["g_fetch", "g_schema", "g_parse", "g_validate"], "stop_at_milestone"),
]


ENVIRONMENT_FIXTURES = {
    "users_basic": {
        "type": "sqlite",
        "setup_sql": [
            "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER, active INTEGER)",
            "INSERT INTO users VALUES (1, 'Ada', 37, 1)",
            "INSERT INTO users VALUES (2, 'Bob', 16, 0)",
            "INSERT INTO users VALUES (3, 'Cara', 22, 1)"
        ],
    },
    "users_orders": {
        "type": "sqlite",
        "setup_sql": [
            "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)",
            "CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)",
            "INSERT INTO users VALUES (1, 'Ada')",
            "INSERT INTO users VALUES (3, 'Cara')",
            "INSERT INTO orders VALUES (101, 1, 50.0)",
            "INSERT INTO orders VALUES (102, 3, 75.5)"
        ],
    },
}


def write_jsonl(path: Path, objects: list[dict]) -> None:
    content = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in objects
    )
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    write_jsonl(ROOT / "skills.jsonl", SKILLS)
    write_jsonl(ROOT / "tasks.jsonl", TASKS)
    write_jsonl(ROOT / "graph_skills.jsonl", GRAPH_SKILLS)
    write_jsonl(ROOT / "graph_tasks.jsonl", GRAPH_TASKS)
    (ROOT / "environment_fixtures.json").write_text(
        json.dumps(ENVIRONMENT_FIXTURES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"generated skills={len(SKILLS)} tasks={len(TASKS)} "
        f"graph_skills={len(GRAPH_SKILLS)} graph_tasks={len(GRAPH_TASKS)}"
    )


if __name__ == "__main__":
    main()
