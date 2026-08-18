"""LLM 调用封装。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .token_utils import estimate_tokens


def _load_project_env() -> None:
    """加载项目根目录 .env，不覆盖系统环境变量。"""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


class LLM:
    SUPPORTED_PROVIDERS = {"deepseek", "mock"}

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        thinking: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        _load_project_env()
        self.temperature = temperature
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if provider not in self.SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(self.SUPPORTED_PROVIDERS))
            raise ValueError(f"不支持的 provider: {provider}；仅支持: {supported}")
        self.provider = provider

        if provider == "deepseek":
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            base_url = base_url or "https://api.deepseek.com"
            if not api_key:
                raise RuntimeError(
                    "未找到 DEEPSEEK_API_KEY。请打开项目根目录的 .env 文件，"
                    "填写：DEEPSEEK_API_KEY=你的API_Key"
                )
            thinking = thinking or "disabled"
            if thinking not in {"enabled", "disabled"}:
                raise ValueError("thinking 必须是 enabled 或 disabled")
            if reasoning_effort is not None and reasoning_effort not in {"high", "max"}:
                raise ValueError("reasoning_effort 必须是 high 或 max")
        self.api_key = api_key
        self.base_url = base_url
        self.thinking = thinking if provider == "deepseek" else None
        self.reasoning_effort = reasoning_effort if provider == "deepseek" else None
        if provider == "deepseek":
            self.model = model or "deepseek-v4-pro"
        else:
            self.model = "mock"

    @property
    def usage(self) -> dict[str, int]:
        """返回当前实例累计 Token 用量。"""
        return dict(self._usage)

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._usage["prompt_tokens"] += int(prompt_tokens)
        self._usage["completion_tokens"] += int(completion_tokens)
        self._usage["total_tokens"] += int(prompt_tokens) + int(completion_tokens)

    def generate(self, messages: list[dict], **kwargs: Any) -> str:
        """生成文本。"""
        if self.provider == "deepseek":
            return self._deepseek(messages, **kwargs)
        return self._mock(messages)

    def generate_json(self, messages: list[dict], **kwargs: Any) -> Any:
        """生成并解析 JSON。"""
        if self.provider == "deepseek":
            kwargs.setdefault("response_format", {"type": "json_object"})
        raw = self.generate(messages, **kwargs)
        return self._parse_json(raw)

    def _deepseek(self, messages: list[dict], **kwargs: Any) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        request_options = self._deepseek_request_options(messages, kwargs)
        resp = client.chat.completions.create(**request_options)
        output = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self._record_usage(
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
            )
        else:
            self._record_estimated_usage(messages, output)
        return output

    def _deepseek_request_options(
        self,
        messages: list[dict],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """构造 DeepSeek OpenAI 兼容请求参数。"""
        options: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        extra_body = dict(options.get("extra_body") or {})
        extra_body["thinking"] = {"type": self.thinking}
        options["extra_body"] = extra_body
        if self.thinking == "enabled":
            options.pop("temperature", None)
            if self.reasoning_effort:
                options.setdefault("reasoning_effort", self.reasoning_effort)
        else:
            options.setdefault("temperature", self.temperature)
            options.pop("reasoning_effort", None)
        return options

    def _mock(self, messages: list[dict]) -> str:
        """生成确定性测试响应。"""
        import re
        last = messages[-1]["content"] if messages else ""

        names = re.findall(
            r"(?m)^\s*(?:\d+\.|[-*])\s*([a-z][a-z0-9_]*)\s*:", last
        )
        skill_names = list(dict.fromkeys(names))

        if "反思" in last:
            output = "检查失败步骤的参数和依赖，重试前先确认错误原因。"
        elif "可用技能" in last or "步骤" in last or "plan" in last.lower():
            if skill_names:
                plan = [
                    {"skill_name": name, "arguments": {}, "reason": "匹配任务描述"}
                    for name in skill_names[:2]
                ]
            else:
                plan = []
            output = json.dumps({"plan": plan}, ensure_ascii=False)
        else:
            output = f"[mock] {last[:80]}"

        self._record_estimated_usage(messages, output)
        return output

    def _record_estimated_usage(self, messages: list[dict], output: str) -> None:
        prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
        self._record_usage(estimate_tokens(prompt_text), estimate_tokens(output))

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """解析 JSON，并兼容 Markdown 代码块。"""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else None
