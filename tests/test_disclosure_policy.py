"""Signal-based disclosure decisions are explicit and label-free at runtime."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.disclosure_policy import SignalDisclosurePolicy
from core.schemas import Skill, Task


def skill(skill_id: str, name: str, required: list[str]) -> Skill:
    return Skill(
        id=skill_id,
        name=name,
        brief_description="处理任务",
        detailed_description="完整行为定义",
        parameters={
            "type": "object",
            "properties": {item: {"type": "string"} for item in required},
            "required": required,
        },
    )


def test_signal_policy_uses_brief_schema_and_full_for_distinct_signals() -> None:
    policy = SignalDisclosurePolicy()
    writer = skill("s1", "write", ["path", "content"])
    patcher = skill("s2", "patch", ["path", "patch"])

    brief = policy.decide(
        Task("t1", "写入文件", inputs={"path": "a", "content": "x"}),
        [writer],
    )
    schema = policy.decide(
        Task(
            "t2",
            "先写入再修改",
            inputs={"path": "a", "content": "x", "patch": "y"},
        ),
        [writer, patcher],
    )
    full = policy.decide(
        Task(
            "t3",
            "原样写入且不自动添加换行",
            inputs={"path": "a", "content": "x"},
        ),
        [writer],
    )

    assert brief.level == "brief"
    assert brief.signals["missing_required_parameters"] == []
    assert schema.level == "schema"
    assert schema.signals["selected_skill_count"] == 2
    assert full.level == "full"
    assert full.signals["matched_behavior_markers"] == ["原样", "不自动"]


if __name__ == "__main__":
    test_signal_policy_uses_brief_schema_and_full_for_distinct_signals()
    print("disclosure policy tests passed")
