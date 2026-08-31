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


if __name__ == "__main__":
    test_mock_call_telemetry_is_phase_aware_and_prompt_safe()
    print("LLM telemetry tests passed")
