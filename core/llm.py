"""LLM 调用封装。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
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
        self._calls: list[dict[str, Any]] = []
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

    @property
    def calls(self) -> list[dict[str, Any]]:
        """返回不含 prompt 正文的逐调用成本和延迟轨迹。"""
        return [dict(item) for item in self._calls]

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._usage["prompt_tokens"] += int(prompt_tokens)
        self._usage["completion_tokens"] += int(completion_tokens)
        self._usage["total_tokens"] += int(prompt_tokens) + int(completion_tokens)

    def generate(self, messages: list[dict], **kwargs: Any) -> str:
        """生成文本。"""
        phase = str(kwargs.pop("_phase", "") or self._infer_phase(messages))
        usage_before = dict(self._usage)
        started = time.perf_counter()
        success = False
        error_type = None
        try:
            if self.provider == "deepseek":
                output = self._deepseek(messages, **kwargs)
            else:
                output = self._mock(messages)
            success = True
            return output
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            usage_delta = {
                key: self._usage[key] - usage_before.get(key, 0)
                for key in self._usage
            }
            self._calls.append({
                "phase": phase,
                "provider": self.provider,
                "model": self.model,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": usage_delta["prompt_tokens"],
                "completion_tokens": usage_delta["completion_tokens"],
                "total_tokens": usage_delta["total_tokens"],
                "message_count": len(messages),
                "prompt_chars": sum(
                    len(str(message.get("content", ""))) for message in messages
                ),
                "success": success,
                "error_type": error_type,
            })

    def generate_json(self, messages: list[dict], **kwargs: Any) -> Any:
        """生成并解析 JSON。"""
        if self.provider == "deepseek":
            kwargs.setdefault("response_format", {"type": "json_object"})
        raw = self.generate(messages, **kwargs)
        try:
            return self._parse_json(raw)
        except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
            preview = raw[:500].replace("\n", "\\n")
            raise ValueError(
                f"LLM 返回的 JSON 无法解析；响应前 500 字符: {preview!r}"
            ) from exc

    def _deepseek(self, messages: list[dict], **kwargs: Any) -> str:
        request_options = self._deepseek_request_options(messages, kwargs)
        try:
            import openai
        except ModuleNotFoundError:
            return self._deepseek_http(messages, request_options)

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(**request_options)
        output = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage:
            self._record_usage(
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
            )
        else:
            self._record_estimated_usage(messages, output)
        return output

    def _deepseek_http(
        self,
        messages: list[dict],
        request_options: dict[str, Any],
    ) -> str:
        """SDK 不可用时使用标准库调用同一兼容接口。"""
        body = dict(request_options)
        body.update(body.pop("extra_body", {}))
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            if shutil.which("curl"):
                payload = self._deepseek_curl(body)
            else:
                raise RuntimeError(f"DeepSeek 网络请求失败: {exc.reason}") from exc

        return self._parse_deepseek_payload(messages, payload)

    def _deepseek_curl(self, body: dict[str, Any]) -> dict[str, Any]:
        """urllib 无法使用系统网络通道时，通过 curl 安全回退。"""
        body_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as stream:
                json.dump(body, stream, ensure_ascii=False)
                body_path = stream.name
            headers = (
                f"Authorization: Bearer {self.api_key}\n"
                "Content-Type: application/json\n"
            )
            completed = subprocess.run(
                [
                    shutil.which("curl") or "curl",
                    "--silent",
                    "--show-error",
                    "--fail-with-body",
                    "--max-time",
                    "120",
                    "--request",
                    "POST",
                    "--header",
                    "@-",
                    "--data-binary",
                    f"@{body_path}",
                    f"{self.base_url.rstrip('/')}/chat/completions",
                ],
                input=headers,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=130,
                check=False,
            )
        finally:
            if body_path:
                Path(body_path).unlink(missing_ok=True)
        if completed.returncode != 0:
            error = (completed.stdout or completed.stderr).strip()
            raise RuntimeError(f"DeepSeek curl 请求失败: {error[:500]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek curl 响应不是有效 JSON") from exc

    def _parse_deepseek_payload(
        self,
        messages: list[dict],
        payload: dict[str, Any],
    ) -> str:
        try:
            output = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek 响应缺少 choices[0].message.content") from exc
        usage = payload.get("usage") or {}
        if usage:
            self._record_usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
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
    def _infer_phase(messages: list[dict]) -> str:
        """根据稳定的 prompt 标记区分选择、规划、修复与反思调用。"""
        text = "\n".join(str(message.get("content", "")) for message in messages)
        if "Schema 校验错误" in text:
            return "argument_repair"
        if "候选技能概览" in text:
            return "skill_selection"
        if "已选技能定义" in text or "可用技能" in text:
            return "argument_planning"
        if "执行轨迹" in text and "反思" in text:
            return "reflection"
        return "unspecified"

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """解析 JSON，并兼容 Markdown 代码块。"""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as direct_error:
            import ast
            import re

            decoder = json.JSONDecoder()
            starts = [raw.find("{")]
            if starts[0] < 0:
                starts = [raw.find("[")]
            for start in starts:
                if start < 0:
                    continue
                candidate = raw[start:]
                try:
                    value, _ = decoder.raw_decode(candidate)
                    return value
                except json.JSONDecodeError:
                    pass
            repaired = re.sub(
                r"(:\s*)(\$(?:last_output|task|input\.[A-Za-z_][\w.]*))"
                r"\s*([,}\]])",
                r'\1"\2"\3',
                raw,
            )
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            if repaired != raw:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raise direct_error
            if isinstance(value, (dict, list)):
                return value
            raise direct_error
