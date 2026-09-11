"""LLM call traces record cost/latency without storing prompt bodies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM


def test_mock_call_telemetry_is_phase_aware_and_prompt_safe() -> None:
    llm = LLM(provider="mock")
    llm.generate_json([{
        "role": "user",
        "content": "候选技能概览：\n- [s1] demo: test\n拆分步骤，不要生成参数。",
    }])
    llm.generate([{
        "role": "user",
        "content": "执行轨迹：失败\n请简要反思。",
    }])

    assert [call["phase"] for call in llm.calls] == [
        "skill_selection",
        "reflection",
    ]
    assert all(call["success"] for call in llm.calls)
    assert all(call["duration_ms"] >= 0 for call in llm.calls)
    assert all(call["total_tokens"] > 0 for call in llm.calls)
    assert all("messages" not in call and "prompt" not in call for call in llm.calls)


def test_one_stage_joint_prompt_is_not_mislabeled_as_selection() -> None:
    llm = LLM(provider="mock")
    llm.generate_json([{
        "role": "user",
        "content": (
            "可用技能：\n第一层：候选技能概览\n"
            "选择完成任务所需的最少技能，并输出 arguments。"
        ),
    }])
    assert llm.calls[-1]["phase"] == "one_stage_joint_planning"


if __name__ == "__main__":
    test_mock_call_telemetry_is_phase_aware_and_prompt_safe()
    test_one_stage_joint_prompt_is_not_mislabeled_as_selection()
    print("LLM telemetry tests passed")
