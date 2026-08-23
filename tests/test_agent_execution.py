"""结构化规划与执行测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import Agent
from core.schemas import RetrievalResult, Skill, Task
from organization.flat import FlatOrganizer


class ScriptedLLM:
    def __init__(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate_json(self, messages):
        return {
            "plan": [
                {"skill_name": "read_file", "arguments": {"path": "input.md"}},
                {
                    "skill_name": "translate",
                    "arguments": {"text": "$last_output", "target_language": "zh"},
                },
            ]
        }

    def generate(self, messages):
        return "执行顺序和参数正确。"


def test_structured_execution():
    skills = [
        Skill(
            id="s1",
            name="read_file",
            brief_description="读取文件",
            detailed_description="读取文本文件",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        Skill(
            id="s2",
            name="translate",
            brief_description="翻译文本",
            detailed_description="翻译到目标语言",
            parameters={"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]},
        ),
    ]
    handlers = {
        "read_file": lambda path: "hello",
        "translate": lambda text, target_language: f"{text}->{target_language}",
    }
    agent = Agent(
        llm=ScriptedLLM(),
        organizer=FlatOrganizer(),
        skill_handlers=handlers,
    )
    task = Task("t1", "读取并翻译", ["s1", "s2"], ["s1", "s2"])
    result = agent.run(task, RetrievalResult(task.instruction, skills, [1.0, 0.8]))

    assert result["success"] is True
    assert result["selected_skill_ids"] == ["s1", "s2"]
    assert result["answer"] == "hello->zh"
    assert result["execution_steps"] == 3


if __name__ == "__main__":
    test_structured_execution()
    print("agent execution test passed")
