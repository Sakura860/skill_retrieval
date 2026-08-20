"""统一 Benchmark 结果结构测试。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.run_benchmark import run_benchmark
from organization.flat import FlatOrganizer
from retrieval.bm25 import BM25Retriever


class ScriptedLLM:
    def __init__(self):
        self.provider = "scripted"
        self.model = "scripted"
        self.temperature = 0.0
        self.thinking = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate_json(self, messages):
        return {
            "plan": [
                {
                    "skill_name": "calculator",
                    "arguments": {"expression": "(35 + 42) * 3"},
                }
            ]
        }

    def generate(self, messages):
        return "not used"


def test_benchmark_returns_layered_serializable_result():
    skills = [{
        "id": "s1",
        "name": "calculator",
        "brief_description": "计算数学表达式",
        "detailed_description": "计算括号和四则运算表达式",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }]
    tasks = [{
        "id": "t1",
        "instruction": "计算 (35 + 42) * 3",
        "expected_skills": ["s1"],
        "expected_skill_sequence": ["s1"],
        "ground_truth": 231,
    }]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        skills_path = root / "skills.json"
        tasks_path = root / "tasks.json"
        skills_path.write_text(json.dumps(skills, ensure_ascii=False), encoding="utf-8")
        tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

        result = run_benchmark(
            skills_path,
            tasks_path,
            retriever=BM25Retriever(),
            organizer=FlatOrganizer(),
            llm=ScriptedLLM(),
            skill_handlers={"calculator": lambda expression: 231},
            top_k=1,
            retrieval_ks=(1,),
            run_id="test-run",
            enable_reflection=False,
        )

    assert result["run_info"]["run_id"] == "test-run"
    assert result["run_info"]["scored_task_count"] == 1
    assert result["metrics"]["retrieval"]["recall@1"] == 1.0
    assert result["metrics"]["agent"]["task_success_rate"] == 1.0
    assert result["metrics"]["efficiency"]["avg_skill_calls"] == 1.0
    assert result["per_task"][0]["retrieved_skill_ids"] == ["s1"]
    assert result["per_task"][0]["execution_success"] is True
    assert result["per_task"][0]["task_success"] is True
    json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    test_benchmark_returns_layered_serializable_result()
    print("benchmark output test passed")
