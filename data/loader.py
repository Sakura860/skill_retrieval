"""加载 Skill 和 Task 数据。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from core.schemas import Skill, Task


def load_skills(path: str | Path) -> list[Skill]:
    """从 JSON、JSONL 或目录加载 Skill。"""
    path = Path(path)
    objs = _read_objs(path)
    skills = [Skill(**_normalize_skill(obj)) for obj in objs]
    _ensure_unique(skills, "id", "Skill ID")
    _ensure_unique(skills, "name", "Skill name")
    return skills


def load_tasks(path: str | Path) -> list[Task]:
    """从 JSON、JSONL 或目录加载 Task。"""
    path = Path(path)
    objs = _read_objs(path)
    tasks = [Task(**obj) for obj in objs]
    _ensure_unique(tasks, "id", "Task ID")
    return tasks


def _read_objs(path: Path) -> list[dict]:
    """读取对象数组，并保留稳定文件顺序。"""
    if path.is_dir():
        objs: list[dict] = []
        for f in sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")):
            objs.extend(_read_objs(f))
        return objs
    if path.suffix == ".jsonl":
        objects = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_number}") from exc
        return objects
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON 顶层必须是数组: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON 数组元素必须是对象: {path}")
    return data


def _normalize_skill(obj: dict) -> dict:
    """兼容旧版 description 和 metadata.depends_on。"""
    data = dict(obj)
    data.pop("source_code", None)
    legacy_description = data.pop("description", "")
    data.setdefault("brief_description", legacy_description)
    data.setdefault("detailed_description", legacy_description)
    metadata = data.get("metadata") or {}
    data.setdefault("dependencies", metadata.get("depends_on", []))
    return data


def _ensure_unique(items: list, attribute: str, label: str) -> None:
    counts = Counter(getattr(item, attribute) for item in items)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} 重复: {', '.join(duplicates)}")
