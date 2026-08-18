"""Token 数量估算。"""
from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def estimate_tokens(text: str) -> int:
    """在无模型 tokenizer 时估算中英文 Token 数。"""
    total = 0
    for token in _TOKEN_RE.findall(text):
        if token.isascii() and token.replace("_", "").isalnum():
            total += max(1, math.ceil(len(token) / 4))
        else:
            total += 1
    return total
