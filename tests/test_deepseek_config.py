"""DeepSeek V4 请求配置测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm import LLM


def test_v4_non_thinking_options():
    llm = LLM(provider="deepseek", api_key="test-key")
    options = llm._deepseek_request_options([{"role": "user", "content": "test"}], {})

    assert llm.model == "deepseek-v4-pro"
    assert options["extra_body"]["thinking"]["type"] == "disabled"
    assert options["temperature"] == 0.0
    assert "reasoning_effort" not in options


def test_v4_thinking_options():
    llm = LLM(
        provider="deepseek",
        api_key="test-key",
        thinking="enabled",
        reasoning_effort="max",
    )
    options = llm._deepseek_request_options(
        [{"role": "user", "content": "test"}],
        {"temperature": 1.0},
    )

    assert options["extra_body"]["thinking"]["type"] == "enabled"
    assert options["reasoning_effort"] == "max"
    assert "temperature" not in options


def test_unsupported_provider():
    try:
        LLM(provider="other", api_key="test-key")
    except ValueError as exc:
        assert "仅支持" in str(exc)
    else:
        raise AssertionError("不支持的 provider 应当报错")


if __name__ == "__main__":
    test_v4_non_thinking_options()
    test_v4_thinking_options()
    test_unsupported_provider()
    print("deepseek config test passed")
