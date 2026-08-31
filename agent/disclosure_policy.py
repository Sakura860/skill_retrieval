"""Auditable disclosure policy derived from development-set failure signals."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.schemas import Skill, Task


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "disclosure_policy_v01.json"
)


@dataclass
class DisclosureDecision:
    level: str
    reason: str
    signals: dict = field(default_factory=dict)


class SignalDisclosurePolicy:
    """Choose the lowest disclosure level from explicit observable signals."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        levels = self.config.get("levels", {})
        if set(levels) != {"brief", "schema", "full"}:
            raise ValueError("disclosure policy 必须定义 brief/schema/full")
        self.behavior_markers = tuple(
            str(item) for item in self.config.get("behavior_markers", [])
        )

    def decide(self, task: Task, selected_skills: list[Skill]) -> DisclosureDecision:
        required_parameters = sorted({
            name
            for skill in selected_skills
            for name in skill.parameters.get("required", [])
            if isinstance(name, str)
        })
        input_keys = sorted(str(key) for key in task.inputs)
        missing_parameters = sorted(set(required_parameters) - set(input_keys))
        matched_markers = [
            marker for marker in self.behavior_markers
            if marker and marker in task.instruction
        ]
        signals = {
            "policy_id": self.config.get("policy_id"),
            "selected_skill_count": len(selected_skills),
            "required_parameters": required_parameters,
            "task_input_keys": input_keys,
            "missing_required_parameters": missing_parameters,
            "matched_behavior_markers": matched_markers,
        }
        if matched_markers:
            return DisclosureDecision(
                level="full",
                reason="observed_non_schema_behavior_constraint",
                signals=signals,
            )
        if len(selected_skills) > 1:
            return DisclosureDecision(
                level="schema",
                reason="multi_skill_dataflow_requires_schema",
                signals=signals,
            )
        if selected_skills and not missing_parameters:
            return DisclosureDecision(
                level="brief",
                reason="named_inputs_cover_single_skill_required_parameters",
                signals=signals,
            )
        return DisclosureDecision(
            level="schema",
            reason="required_parameter_mapping_is_incomplete",
            signals=signals,
        )
