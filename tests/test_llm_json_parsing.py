"""Tolerant parsing for common, unambiguous model JSON wrappers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM


def test_parse_json_wrappers_and_common_safe_repairs() -> None:
    assert LLM._parse_json('```json\n{"plan": []}\n```') == {"plan": []}
    assert LLM._parse_json('result:\n{"plan": []}\nthanks') == {"plan": []}
    assert LLM._parse_json('{"plan": [],}') == {"plan": []}
    assert LLM._parse_json("{'plan': [], 'ok': True}") == {
        "plan": [], "ok": True,
    }
    assert LLM._parse_json(
        '{"arguments": {"left": $input.left, "data": $last_output}}'
    ) == {
        "arguments": {"left": "$input.left", "data": "$last_output"}
    }


if __name__ == "__main__":
    test_parse_json_wrappers_and_common_safe_repairs()
    print("LLM JSON parsing tests passed")
