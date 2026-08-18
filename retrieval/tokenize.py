"""中英文轻量分词。"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_HAN_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = _WORD_RE.findall(text)

    han = _HAN_RE.findall(text)
    tokens += [han[i] + han[i + 1] for i in range(len(han) - 1)]
    tokens += han
    return tokens
