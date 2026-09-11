"""Pinned adapter for the official Graph-of-Skills reverse-PPR runtime."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

from core.schemas import Skill, Task
from .base import BaseOrganizer, OrganizedContext, append_whole_block


OFFICIAL_COMMIT = "203f60a2c689da055ce1ac351eb3cb9912a3bca7"
OFFICIAL_QUERY_SHA256 = "0b47d795f662e2954ac70f949942d9345dca1f2a011f1dbe1fae22900909eccb"
OFFICIAL_QUERY_PATH = Path("evaluation/skillsbench/graphskills_assets/query.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_runtime(upstream_root: str | Path) -> ModuleType:
    """Load the exact pinned self-contained retrieval runtime after provenance checks."""
    root = Path(upstream_root).resolve()
    query_path = root / OFFICIAL_QUERY_PATH
    if not query_path.is_file():
        raise FileNotFoundError(f"Graph-of-Skills official runtime not found: {query_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"Graph-of-Skills commit mismatch: expected {OFFICIAL_COMMIT}, got {commit}"
        )
    actual_hash = _sha256(query_path)
    if actual_hash != OFFICIAL_QUERY_SHA256:
        raise RuntimeError(
            "Graph-of-Skills query.py hash mismatch: "
            f"expected {OFFICIAL_QUERY_SHA256}, got {actual_hash}"
        )
    spec = importlib.util.spec_from_file_location(
        "skill_agent_pinned_gos_query",
        query_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Graph-of-Skills runtime: {query_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialGoSReversePPROrganizer(BaseOrganizer):
    """Apply official reverse-aware PPR to fixed candidate seeds and task edges.

    This is the official lexical-ablation structural core with externally fixed
    seeds. It is intentionally not labeled as the full hybrid GoS pipeline.
    """

    def __init__(
        self,
        catalog: list[Skill],
        upstream_root: str | Path,
        *,
        max_additional_skills: int = 2,
        include_dataflow: bool = False,
        ppr_damping: float = 0.2,
        ppr_max_iter: int = 50,
        ppr_tolerance: float = 1e-6,
    ):
        if max_additional_skills < 0:
            raise ValueError("max_additional_skills 不能小于 0")
        self.catalog = list(catalog)
        self.by_id = {skill.id: skill for skill in catalog}
        if len(self.by_id) != len(catalog):
            raise ValueError("GoS catalog 中存在重复 Skill ID")
        self.runtime = load_official_runtime(upstream_root)
        self.upstream_root = str(Path(upstream_root).resolve())
        self.max_additional_skills = max_additional_skills
        self.include_dataflow = include_dataflow
        self.ppr_damping = ppr_damping
        self.ppr_max_iter = ppr_max_iter
        self.ppr_tolerance = ppr_tolerance

    def organize(self, skills: list[Skill], task: Task | None = None) -> str:
        return self.organize_context(skills, task).text

    def organize_context(
        self,
        skills: list[Skill],
        task: Task | None = None,
        context_budget_tokens: int | None = None,
    ) -> OrganizedContext:
        if task is None:
            raise ValueError("Official GoS adapter requires task graph_edges")
        if context_budget_tokens is not None and context_budget_tokens < 1:
            raise ValueError("context_budget_tokens 必须大于 0")
        original_ids = list(dict.fromkeys(skill.id for skill in skills))
        unknown = [skill_id for skill_id in original_ids if skill_id not in self.by_id]
        if unknown:
            raise ValueError("GoS seeds contain unknown Skill IDs: " + ", ".join(unknown))

        official_skills = [self._official_skill(skill) for skill in self.catalog]
        official_edges = self._official_edges(task)
        name_to_index = {
            skill["name"]: index for index, skill in enumerate(official_skills)
        }
        seed_indices = [name_to_index[self.by_id[skill_id].name] for skill_id in original_ids]
        seed_weights = self.runtime.build_rank_distribution(len(seed_indices))
        personalization = [0.0] * len(official_skills)
        for index, weight in zip(seed_indices, seed_weights):
            personalization[index] += float(weight)
        transition = self.runtime.build_transition(
            official_skills,
            official_edges,
            reverse_mode="full",
        )
        scores = self.runtime.personalized_pagerank(
            transition,
            personalization,
            damping=self.ppr_damping,
            max_iter=self.ppr_max_iter,
            tol=self.ppr_tolerance,
        )
        maximum = min(
            len(self.catalog),
            len(original_ids) + self.max_additional_skills,
        )
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )[:maximum]
        selected = [self.catalog[index] for index in ranked_indices]
        selected_ids = [skill.id for skill in selected]
        original_set = set(original_ids)
        added_ids = [skill_id for skill_id in selected_ids if skill_id not in original_set]
        dropped_ids = [skill_id for skill_id in original_ids if skill_id not in selected_ids]

        blocks: list[str] = []
        header = (
            "可用技能（官方 Graph-of-Skills reverse-PPR 排名；固定外部 seeds）："
        )
        header_added = append_whole_block(blocks, header, context_budget_tokens)
        exposed: list[str] = []
        truncated: list[str] = []
        for index, skill in enumerate(selected):
            if not header_added:
                truncated.append(skill.id)
                continue
            if append_whole_block(
                blocks,
                f"- {skill.to_prompt(detailed=True)}",
                context_budget_tokens,
            ):
                exposed.append(skill.id)
            else:
                truncated.extend(item.id for item in selected[index:])
                break
        relation_lines = [
            f"- {edge['source']} -> {edge['target']} ({edge['type']}): {edge['description']}"
            for edge in official_edges
            if edge["source"] in {self.by_id[item].name for item in exposed}
            and edge["target"] in {self.by_id[item].name for item in exposed}
        ]
        if relation_lines:
            append_whole_block(
                blocks,
                "图关系：\n" + "\n".join(relation_lines),
                context_budget_tokens,
            )
        issues = []
        if dropped_ids:
            issues.append("gos_dropped_original_seeds:" + ",".join(dropped_ids))
        return OrganizedContext(
            text="\n".join(blocks),
            exposed_skill_ids=exposed,
            detailed_skill_ids=list(exposed),
            truncated_skill_ids=truncated,
            added_skill_ids=added_ids,
            graph_issues=issues,
            resolved_skills=selected,
            context_budget_tokens=context_budget_tokens,
        )

    @staticmethod
    def _official_skill(skill: Skill) -> dict[str, Any]:
        properties = skill.parameters.get("properties", {})
        return {
            "name": skill.name,
            "description": skill.detailed_description,
            "inputs": sorted(properties),
            "outputs": [str(skill.returns.get("type", ""))],
            "rendered_snippet": skill.to_prompt(level="full"),
            "source_path": f"skill-agent://{skill.id}",
            "raw_content": skill.to_prompt(level="full"),
        }

    def _official_edges(self, task: Task) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in task.metadata.get("graph_edges", []):
            if not isinstance(item, dict):
                continue
            relation = str(item.get("type", item.get("relation_type", "")))
            if relation == "prerequisite":
                official_type = "dependency"
            elif relation == "dataflow" and self.include_dataflow:
                official_type = "workflow"
            else:
                continue
            source = self.by_id.get(str(item.get("source_id", "")))
            target = self.by_id.get(str(item.get("target_id", "")))
            if source is None or target is None:
                continue
            output.append({
                "source": source.name,
                "target": target.name,
                "type": official_type,
                "description": str(item.get("evidence", relation)),
                "weight": float(item.get("weight", 1.0)),
                "confidence": 1.0,
            })
        return output

    def provenance(self) -> dict[str, Any]:
        return {
            "paper": "Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills",
            "paper_url": "https://arxiv.org/abs/2604.05333",
            "repository": "https://github.com/davidliuk/graph-of-skills",
            "upstream_commit": OFFICIAL_COMMIT,
            "runtime_path": OFFICIAL_QUERY_PATH.as_posix(),
            "runtime_sha256": OFFICIAL_QUERY_SHA256,
            "component": "official reverse-aware PPR runtime with fixed external seeds",
            "full_hybrid_pipeline": False,
            "include_dataflow_as_workflow": self.include_dataflow,
            "ppr_damping": self.ppr_damping,
            "ppr_max_iter": self.ppr_max_iter,
            "ppr_tolerance": self.ppr_tolerance,
        }
