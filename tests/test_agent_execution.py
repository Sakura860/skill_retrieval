"""结构化规划与执行测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import Agent
from agent.schema_validation import validate_last_output_references
from core.schemas import RetrievalResult, Skill, Task
from organization.flat import FlatOrganizer
from organization.hierarchical import HierarchicalOrganizer


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


class RepairingLLM:
    def __init__(self):
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {"skill_ids": ["s_extract"]}
        if self.calls == 2:
            return {
                "plan": [{
                    "skill_name": "extract_fields",
                    "arguments": {"input": {"name": "Ada"}},
                }]
            }
        return {
            "plan": [{
                "skill_name": "extract_fields",
                "arguments": {"data": {"name": "Ada"}, "fields": ["name"]},
            }]
        }

    def generate(self, messages):
        return "not used"


class AlwaysInvalidLLM(RepairingLLM):
    def generate_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {"skill_ids": ["s_write"]}
        return {
            "plan": [{
                "skill_name": "write_value",
                "arguments": {"wrong_name": 3},
            }]
        }


class AdaptiveRepairingLLM(RepairingLLM):
    def generate_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "selections": [{
                    "requirement": "写入值",
                    "skill_id": "s_adaptive",
                    "information_need": "brief",
                    "need_reason": "操作看起来直接",
                }]
            }
        if self.calls == 2:
            return {
                "plan": [{
                    "skill_name": "write_adaptive",
                    "arguments": {"wrong_name": 3},
                }]
            }
        return {
            "plan": [{
                "skill_name": "write_adaptive",
                "arguments": {"value": 3},
            }]
        }


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


def test_two_stage_planner_discloses_selected_schema_and_repairs_once():
    llm = RepairingLLM()
    skill = Skill(
        id="s_extract",
        name="extract_fields",
        brief_description="从对象保留字段",
        detailed_description="data 是输入对象，fields 是要保留的字段名。",
        parameters={
            "type": "object",
            "properties": {
                "data": {"type": "object"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["data", "fields"],
            "additionalProperties": False,
        },
    )
    agent = Agent(
        llm=llm,
        organizer=HierarchicalOrganizer(detail_top_k=0),
        skill_handlers={
            "extract_fields": lambda data, fields: {
                key: data[key] for key in fields
            }
        },
        enable_reflection=False,
        planner_mode="two_stage",
        max_argument_repairs=1,
    )
    task = Task("t2", "从对象中提取 name", [skill.id], [skill.id])
    result = agent.run(task, RetrievalResult(task.instruction, [skill], [1.0]))

    assert result["success"] is True
    assert result["answer"] == "{'name': 'Ada'}"
    assert result["detailed_skill_ids"] == [skill.id]
    assert result["planner_mode"] == "two_stage"
    assert result["planner_calls"] == 3
    assert result["repair_attempts"] == 1
    assert result["validation_errors"] == []
    assert result["selection_context_tokens"] > 0
    assert result["planning_context_tokens"] > 0
    assert result["skill_context_tokens"] == (
        result["selection_context_tokens"] + result["planning_context_tokens"]
    )
    assert llm.calls == 3


def test_two_stage_planner_never_executes_a_plan_that_stays_invalid():
    llm = AlwaysInvalidLLM()
    executions = []
    skill = Skill(
        id="s_write",
        name="write_value",
        brief_description="写入一个值",
        detailed_description="value 是必填整数。",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    agent = Agent(
        llm=llm,
        organizer=HierarchicalOrganizer(detail_top_k=0),
        skill_handlers={"write_value": lambda value: executions.append(value)},
        enable_reflection=False,
        planner_mode="two_stage",
    )
    task = Task("t3", "写入整数 3", [skill.id], [skill.id])
    result = agent.run(task, RetrievalResult(task.instruction, [skill], [1.0]))

    assert result["success"] is False
    assert result["selected_skill_ids"] == []
    assert result["failure_reason"] == "schema_validation_failed"
    assert result["repair_attempts"] == 1
    assert result["validation_errors"]
    assert executions == []


def test_last_output_schema_rejects_row_arrays_as_json_object_records():
    errors = validate_last_output_references(
        {"records": "$last_output"},
        {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
        },
        {"type": "array", "items": {"type": "array"}},
    )
    compatible = validate_last_output_references(
        {"records": "$last_output"},
        {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
        },
        {"type": "array", "items": {"type": "object"}},
    )

    assert any("前一步返回 array，参数要求 object" in error for error in errors)
    assert compatible == []


def test_skill_prompt_has_separate_brief_schema_and_full_levels():
    skill = Skill(
        id="s_levels",
        name="write_exact",
        brief_description="写入文本",
        detailed_description="必须逐字写入并保留结尾换行。",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        examples=["write_exact(text='hello\\n')"],
    )
    brief = skill.to_prompt(level="brief")
    schema = skill.to_prompt(level="schema")
    full = skill.to_prompt(level="full")

    assert "参数Schema" not in brief
    assert "参数Schema" in schema
    assert "逐字写入" not in schema
    assert "逐字写入" in full
    assert "示例" in full
    assert len(brief) < len(schema) < len(full)


def test_adaptive_disclosure_escalates_only_after_hidden_schema_failure():
    llm = AdaptiveRepairingLLM()
    skill = Skill(
        id="s_adaptive",
        name="write_adaptive",
        brief_description="写入值",
        detailed_description="写入一个整数值。",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    agent = Agent(
        llm=llm,
        organizer=HierarchicalOrganizer(detail_top_k=0),
        skill_handlers={"write_adaptive": lambda value: value},
        enable_reflection=False,
        planner_mode="two_stage",
        planner_disclosure_level="adaptive",
    )
    task = Task("t_adaptive", "写入整数 3", [skill.id], [skill.id])
    result = agent.run(task, RetrievalResult(task.instruction, [skill], [1.0]))

    assert result["success"] is True
    assert result["planner_disclosure_level"] == "adaptive"
    assert result["resolved_disclosure_level"] == "schema"
    assert result["requested_disclosure_levels"] == ["brief"]
    assert result["escalation_reasons"] == ["hidden_schema_validation_failed"]
    assert result["repair_attempts"] == 1


if __name__ == "__main__":
    test_structured_execution()
    test_two_stage_planner_discloses_selected_schema_and_repairs_once()
    test_two_stage_planner_never_executes_a_plan_that_stays_invalid()
    test_last_output_schema_rejects_row_arrays_as_json_object_records()
    test_skill_prompt_has_separate_brief_schema_and_full_levels()
    test_adaptive_disclosure_escalates_only_after_hidden_schema_failure()
    print("agent execution test passed")
