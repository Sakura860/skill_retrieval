"""YAML 配置加载。"""
from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("缺少 PyYAML，请运行: pip install -r requirements.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """加载项目配置。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象: {config_path}")
    return data
