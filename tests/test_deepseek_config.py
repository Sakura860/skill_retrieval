"""DeepSeek V4 请求配置测试。"""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

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


def test_standard_library_http_fallback():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"plan": []}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            }).encode("utf-8")

    llm = LLM(provider="deepseek", api_key="test-key")
    options = llm._deepseek_request_options(
        [{"role": "user", "content": "test"}],
        {"response_format": {"type": "json_object"}},
    )
    with patch("urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
        output = llm._deepseek_http(
            [{"role": "user", "content": "test"}],
            options,
        )

    sent = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
    assert output == '{"plan": []}'
    assert sent["thinking"] == {"type": "disabled"}
    assert "extra_body" not in sent
    assert llm.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


def test_curl_fallback_after_urllib_network_error():
    payload = json.dumps({
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    })
    llm = LLM(provider="deepseek", api_key="test-key")
    options = llm._deepseek_request_options(
        [{"role": "user", "content": "test"}],
        {},
    )
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("unreachable"),
        ),
        patch("shutil.which", return_value="curl"),
        patch(
            "subprocess.run",
            return_value=CompletedProcess([], 0, payload, ""),
        ) as run,
    ):
        output = llm._deepseek_http(
            [{"role": "user", "content": "test"}],
            options,
        )

    assert output == "ok"
    assert "Authorization: Bearer test-key" in run.call_args.kwargs["input"]
    assert "test-key" not in " ".join(run.call_args.args[0])
    assert llm.usage["total_tokens"] == 4


if __name__ == "__main__":
    test_v4_non_thinking_options()
    test_v4_thinking_options()
    test_unsupported_provider()
    test_standard_library_http_fallback()
    test_curl_fallback_after_urllib_network_error()
    print("deepseek config test passed")
