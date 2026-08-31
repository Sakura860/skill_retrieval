"""生成 body-aware、staged planning 与 typed graph 定向数据 v0.2。"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
V01 = ROOT.parent / "benchmark_v01"
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_skills  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402


def exact(expected: Any) -> dict:
    return {"verifier_type": "exact_match", "expected_output": expected}


def json_match(expected: Any) -> dict:
    return {"verifier_type": "json_match", "expected_output": expected}


def file_state(files: dict, required: list[str], forbidden: list[str]) -> dict:
    return {
        "verifier_type": "file_state",
        "expected_state": {"files": files},
        "required_effects": required,
        "forbidden_effects": forbidden,
    }


def sqlite_state(query: str, rows: list) -> dict:
    return {
        "verifier_type": "sqlite_state",
        "expected_state": {
            "query": query,
            "params": [],
            "expected_rows": rows,
            "ordered": True,
        },
    }


SPECS = [
    {
        "id": "v2body01", "split": "dev", "family": "file_operation",
        "template_id": "body_create_nonexistent",
        "instruction": "仅当目标不存在时创建 UTF-8 文件 artifact.txt，不能覆盖已有文件。",
        "sequence": ["sfile_create"],
        "inputs": {"path": "artifact.txt", "content": "alpha"},
        "evaluation": file_state(
            {"artifact.txt": {"exists": True, "content": "alpha"}},
            ["artifact.txt"], [],
        ),
        "environment": {"initial_state": {"files": {}}},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2body02", "split": "test", "family": "file_operation",
        "template_id": "body_overwrite_existing",
        "instruction": "只覆盖已有 status.txt 的全部内容，不允许创建新文件或追加。",
        "sequence": ["sfile_overwrite"],
        "inputs": {"path": "status.txt", "content": "approved"},
        "evaluation": file_state(
            {"status.txt": {"exists": True, "content": "approved"}},
            ["status.txt"], [],
        ),
        "environment": {"initial_state": {"files": {
            "status.txt": {"exists": True, "content": "pending"},
        }}},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2body03", "split": "dev", "family": "file_operation",
        "template_id": "body_append_exact",
        "instruction": "在已有 log.txt 末尾原样追加 content，不自动添加换行。",
        "sequence": ["sfile_append"],
        "inputs": {"path": "log.txt", "content": "\nnext"},
        "evaluation": file_state(
            {"log.txt": {"exists": True, "content": "base\nnext"}},
            ["log.txt"], [],
        ),
        "environment": {"initial_state": {"files": {
            "log.txt": {"exists": True, "content": "base"},
        }}},
        "questions": ["body_disambiguation", "low_initial_rank"],
    },
    {
        "id": "v2body04", "split": "dev", "family": "json_text",
        "template_id": "body_merge_policy",
        "instruction": "合并两个 JSON 对象，键冲突时 conflict_policy 使用 right。",
        "sequence": ["sjson_merge"],
        "inputs": {
            "left": {"theme": "light", "lang": "zh"},
            "right": {"theme": "dark"},
            "conflict_policy": "right",
        },
        "evaluation": json_match({"theme": "dark", "lang": "zh"}),
        "environment": {},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2body05", "split": "test", "family": "json_text",
        "template_id": "body_filter_not_sort",
        "instruction": "对 JSON 对象数组按 active 字段等值筛选，不排序。",
        "sequence": ["sjson_filter"],
        "inputs": {
            "records": [
                {"name": "Ada", "active": True},
                {"name": "Bob", "active": False},
            ],
            "field": "active", "equals": True,
        },
        "evaluation": json_match([{"name": "Ada", "active": True}]),
        "environment": {},
        "questions": ["body_disambiguation", "low_initial_rank"],
    },
    {
        "id": "v2body06", "split": "dev", "family": "sqlite",
        "template_id": "body_aggregate_not_rows",
        "instruction": "执行 COUNT 聚合查询，只返回统计值而不是逐行明细。",
        "sequence": ["sdb_aggregate"],
        "inputs": {"query": "SELECT COUNT(*) FROM users WHERE age >= 18"},
        "evaluation": sqlite_state(
            "SELECT COUNT(*) FROM users WHERE age >= 18", [[2]],
        ),
        "environment": {"sqlite_fixture_id": "users_basic"},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2rank01", "split": "dev", "family": "sqlite",
        "template_id": "low_rank_update_not_delete",
        "instruction": "把 Bob 的 active 改为 1，不插入也不删除记录。",
        "sequence": ["sdb_update"],
        "inputs": {
            "table": "users", "where": {"name": "Bob"},
            "changes": {"active": 1},
        },
        "evaluation": sqlite_state(
            "SELECT active FROM users WHERE name='Bob'", [[1]],
        ),
        "environment": {"sqlite_fixture_id": "users_basic"},
        "questions": ["low_initial_rank", "alternative_conflict"],
        "graph_edges": [
            {"source_id": "sdb_update", "target_id": "sdb_delete", "type": "alternative", "evidence": "same record-selection intent"},
            {"source_id": "sdb_update", "target_id": "sdb_delete", "type": "conflict", "evidence": "mutate versus remove"},
        ],
    },
    {
        "id": "v2rank02", "split": "dev", "family": "calculation",
        "template_id": "low_rank_schema_mean",
        "instruction": "对 values 执行 operation=mean，返回算术平均值。",
        "sequence": ["scalc_agg"],
        "inputs": {"values": [4, 9, 17], "operation": "mean"},
        "evaluation": exact(10),
        "environment": {},
        "questions": ["low_initial_rank", "schema_repair"],
        "expected_initial_schema_error": "missing_required_parameters",
    },
    {
        "id": "v2graph01", "split": "dev", "family": "file_operation",
        "template_id": "graph_copy_then_append",
        "instruction": "先复制 source.txt 为 backup.txt 并保留源文件，再在 backup.txt 末尾追加一行 extra。",
        "sequence": ["sfile_copy", "sfile_append"],
        "inputs": {
            "source": "source.txt", "destination": "backup.txt",
            "append_content": "\nextra",
        },
        "evaluation": file_state(
            {
                "source.txt": {"exists": True, "content": "base"},
                "backup.txt": {"exists": True, "content": "base\nextra"},
            },
            ["backup.txt"], ["source.txt"],
        ),
        "environment": {"initial_state": {"files": {
            "source.txt": {"exists": True, "content": "base"},
        }}},
        "questions": ["multi_skill_graph", "prerequisite_completion", "dataflow"],
        "omit_from_candidates": ["sfile_copy"],
        "graph_edges": [
            {"source_id": "sfile_copy", "target_id": "sfile_append", "type": "prerequisite", "evidence": "backup.txt must exist before append"},
            {"source_id": "sfile_copy", "target_id": "sfile_append", "type": "dataflow", "evidence": "destination path -> path"},
        ],
    },
    {
        "id": "v2graph02", "split": "dev", "family": "file_operation",
        "template_id": "graph_create_then_patch",
        "instruction": "先创建 settings.json，再只把其中 theme 修改为 dark 并保留 lang。",
        "sequence": ["sfile_create", "sfile_patch_json"],
        "inputs": {
            "path": "settings.json",
            "initial_content": "{\"theme\":\"light\",\"lang\":\"zh\"}",
            "patch": {"theme": "dark"},
        },
        "evaluation": file_state(
            {"settings.json": {"exists": True, "json": {
                "theme": "dark", "lang": "zh",
            }}},
            ["settings.json"], [],
        ),
        "environment": {"initial_state": {"files": {}}},
        "questions": ["multi_skill_graph", "prerequisite_completion", "dataflow"],
        "omit_from_candidates": ["sfile_create"],
        "graph_edges": [
            {"source_id": "sfile_create", "target_id": "sfile_patch_json", "type": "prerequisite", "evidence": "settings.json must exist before patch"},
            {"source_id": "sfile_create", "target_id": "sfile_patch_json", "type": "dataflow", "evidence": "created path -> path"},
        ],
    },
    {
        "id": "v2alt01", "split": "test", "family": "file_operation",
        "template_id": "alternative_copy_not_move",
        "instruction": "复制 source.txt 到 archive.txt 并保留源文件，不要移动。",
        "sequence": ["sfile_copy"],
        "inputs": {"source": "source.txt", "destination": "archive.txt"},
        "evaluation": file_state(
            {
                "source.txt": {"exists": True, "content": "payload"},
                "archive.txt": {"exists": True, "content": "payload"},
            },
            ["archive.txt"], ["source.txt"],
        ),
        "environment": {"initial_state": {"files": {
            "source.txt": {"exists": True, "content": "payload"},
        }}},
        "questions": ["alternative_conflict"],
        "graph_edges": [
            {"source_id": "sfile_copy", "target_id": "sfile_move", "type": "alternative", "evidence": "same source/destination shape"},
            {"source_id": "sfile_copy", "target_id": "sfile_move", "type": "conflict", "evidence": "preserve versus remove source"},
        ],
    },
    {
        "id": "v2alt02", "split": "dev", "family": "sqlite",
        "template_id": "conflict_delete_not_update",
        "instruction": "删除 active=0 的用户记录，不要把它们更新为 active=1。",
        "sequence": ["sdb_delete"],
        "inputs": {"table": "users", "where": {"active": 0}},
        "evaluation": sqlite_state(
            "SELECT name FROM users ORDER BY name", [["Ada"], ["Cara"]],
        ),
        "environment": {"sqlite_fixture_id": "users_basic"},
        "questions": ["alternative_conflict"],
        "graph_edges": [
            {"source_id": "sdb_delete", "target_id": "sdb_update", "type": "alternative", "evidence": "same predicate scope"},
            {"source_id": "sdb_delete", "target_id": "sdb_update", "type": "conflict", "evidence": "delete versus mutate"},
        ],
    },
    {
        "id": "v2schema01", "split": "dev", "family": "json_text",
        "template_id": "schema_merge_required_fields",
        "instruction": "合并 left 与 right，冲突时保留 left 的值。",
        "sequence": ["sjson_merge"],
        "inputs": {
            "left": {"theme": "light", "lang": "zh"},
            "right": {"theme": "dark", "region": "CN"},
            "conflict_policy": "left",
        },
        "evaluation": json_match({
            "theme": "light", "lang": "zh", "region": "CN",
        }),
        "environment": {},
        "questions": ["schema_repair"],
        "expected_initial_schema_error": "wrong_parameter_names",
    },
    {
        "id": "v2schema02", "split": "test", "family": "calculation",
        "template_id": "schema_unit_required_fields",
        "instruction": "把 value 从 from_unit 换算为 to_unit。",
        "sequence": ["scalc_unit"],
        "inputs": {"value": 7.5, "from_unit": "公里", "to_unit": "米"},
        "evaluation": exact(7500),
        "environment": {},
        "questions": ["schema_repair"],
        "expected_initial_schema_error": "missing_required_parameters",
    },
    {
        "id": "v2confirm01", "split": "test", "family": "file_operation",
        "template_id": "confirm_append_preserve_prefix",
        "instruction": "保留 notes.txt 的原内容，并把 suffix 原样追加到文件末尾。",
        "sequence": ["sfile_append"],
        "inputs": {"path": "notes.txt", "content": "tail"},
        "evaluation": file_state(
            {"notes.txt": {"exists": True, "content": "head\ntail"}},
            ["notes.txt"], [],
        ),
        "environment": {"initial_state": {"files": {
            "notes.txt": {"exists": True, "content": "head\n"},
        }}},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2confirm02", "split": "test", "family": "json_text",
        "template_id": "confirm_rename_preserve_unmapped",
        "instruction": "把 JSON 对象的 first 键改名为 given_name，并保留所有未映射字段。",
        "sequence": ["sjson_rename"],
        "inputs": {
            "data": {"first": "Ada", "role": "admin"},
            "mapping": {"first": "given_name"},
        },
        "evaluation": json_match({"given_name": "Ada", "role": "admin"}),
        "environment": {},
        "questions": ["body_disambiguation", "low_initial_rank"],
    },
    {
        "id": "v2confirm03", "split": "test", "family": "sqlite",
        "template_id": "confirm_sqlite_sum_not_rows",
        "instruction": "在 users 表上执行 SUM(active) 聚合，只返回聚合结果。",
        "sequence": ["sdb_aggregate"],
        "inputs": {"query": "SELECT SUM(active) FROM users"},
        "evaluation": sqlite_state(
            "SELECT SUM(active) FROM users", [[2]],
        ),
        "environment": {"sqlite_fixture_id": "users_basic"},
        "questions": ["body_disambiguation"],
    },
    {
        "id": "v2confirm04", "split": "test", "family": "calculation",
        "template_id": "confirm_linear_equation_schema",
        "instruction": "求一元一次方程 a*x+b=c 的 x。",
        "sequence": ["scalc_eq"],
        "inputs": {"a": 2, "b": 3, "c": 13},
        "evaluation": exact(5),
        "environment": {},
        "questions": ["schema_repair", "low_initial_rank"],
        "expected_initial_schema_error": "missing_required_parameters",
    },
    {
        "id": "v2confirm05", "split": "test", "family": "json_text",
        "template_id": "confirm_extract_then_rename",
        "instruction": "先从 data 只保留 first 和 last，再把 first 重命名为 given_name。",
        "sequence": ["sjson_extract", "sjson_rename"],
        "inputs": {
            "data": {"first": "Ada", "last": "Lovelace", "age": 36},
            "fields": ["first", "last"],
            "mapping": {"first": "given_name"},
        },
        "evaluation": json_match({
            "given_name": "Ada", "last": "Lovelace",
        }),
        "environment": {},
        "questions": ["multi_skill_graph", "prerequisite_completion", "dataflow"],
        "omit_from_candidates": ["sjson_extract"],
        "graph_edges": [
            {"source_id": "sjson_extract", "target_id": "sjson_rename", "type": "prerequisite", "evidence": "unwanted fields must be removed before renaming"},
            {"source_id": "sjson_extract", "target_id": "sjson_rename", "type": "dataflow", "evidence": "object output -> data object"},
        ],
    },
    {
        "id": "v2confirm06", "split": "test", "family": "sqlite",
        "template_id": "confirm_update_preserve_rows",
        "instruction": "把 Bob 的 active 更新为 1，必须保留 Bob 和其他所有记录，不能删除。",
        "sequence": ["sdb_update"],
        "inputs": {
            "table": "users", "where": {"name": "Bob"},
            "changes": {"active": 1},
        },
        "evaluation": sqlite_state(
            "SELECT name, active FROM users ORDER BY name",
            [["Ada", 1], ["Bob", 1], ["Cara", 1]],
        ),
        "environment": {"sqlite_fixture_id": "users_basic"},
        "questions": ["alternative_conflict"],
        "graph_edges": [
            {"source_id": "sdb_update", "target_id": "sdb_delete", "type": "alternative", "evidence": "same predicate scope"},
            {"source_id": "sdb_update", "target_id": "sdb_delete", "type": "conflict", "evidence": "preserve and mutate versus remove"},
        ],
    },
]


def _write_jsonl(path: Path, objects: list[dict]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in objects
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    skills = load_skills(V01 / "skills.jsonl")
    brief = BM25Retriever(text_level="brief")
    detailed = BM25Retriever(text_level="detailed")
    all_field = BM25Retriever(text_level="all")
    for retriever in (brief, detailed, all_field):
        retriever.index(skills)
    skill_ids = [skill.id for skill in skills]
    tasks = []
    for spec in SPECS:
        rankings = {
            "brief": brief.retrieve(spec["instruction"], len(skills)).ranked_ids(),
            "detailed": detailed.retrieve(spec["instruction"], len(skills)).ranked_ids(),
            "all": all_field.retrieve(spec["instruction"], len(skills)).ranked_ids(),
        }
        sequence = spec["sequence"]
        candidates = rankings["brief"][:10]
        for omitted in spec.get("omit_from_candidates", []):
            candidates = [item for item in candidates if item != omitted]
        for gold_id in sequence:
            if gold_id not in candidates and gold_id not in spec.get("omit_from_candidates", []):
                candidates[-1] = gold_id
        candidates = list(dict.fromkeys(candidates))
        tasks.append({
            "id": spec["id"],
            "instruction": spec["instruction"],
            "inputs": spec["inputs"],
            "expected_skills": list(dict.fromkeys(sequence)),
            "expected_skill_sequence": sequence,
            "ground_truth": None,
            "metadata": {
                "dataset": "benchmark_v02",
                "split": spec["split"],
                "family": spec["family"],
                "template_id": spec["template_id"],
                "variant_id": 1,
                "step_count": len(sequence),
                "research_questions": spec["questions"],
                "candidate_skill_ids": candidates,
                "candidate_source": "brief_bm25_top10_with_gold_control",
                "context_budget_tokens": 3200,
                "raw_ranks": {
                    level: {gold_id: ranking.index(gold_id) + 1 for gold_id in sequence}
                    for level, ranking in rankings.items()
                },
                "graph_edges": spec.get("graph_edges", []),
                "intentionally_omitted_skill_ids": spec.get(
                    "omit_from_candidates", []
                ),
                "expected_initial_schema_error": spec.get(
                    "expected_initial_schema_error"
                ),
                "environment_fixture": spec["environment"],
            },
            "evaluation": spec["evaluation"],
        })

    unknown = {
        skill_id
        for task in tasks
        for skill_id in task["expected_skills"]
        if skill_id not in skill_ids
    }
    if unknown:
        raise ValueError(f"任务引用未知 Skill: {', '.join(sorted(unknown))}")
    shutil.copyfile(V01 / "skills.jsonl", ROOT / "skills.jsonl")
    shutil.copyfile(
        V01 / "environment_fixtures.json",
        ROOT / "environment_fixtures.json",
    )
    _write_jsonl(ROOT / "tasks.jsonl", tasks)
    print(
        f"generated skills={len(skills)} tasks={len(tasks)} "
        f"dev={sum(task['metadata']['split'] == 'dev' for task in tasks)} "
        f"test={sum(task['metadata']['split'] == 'test' for task in tasks)}"
    )


if __name__ == "__main__":
    main()
