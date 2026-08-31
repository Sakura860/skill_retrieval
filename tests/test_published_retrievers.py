"""Published-baseline protocol tests that do not download model weights."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.schemas import RetrievalResult, Skill
from retrieval.published import (
    SKILLROUTER_ENCODER_MODEL,
    SKILLROUTER_UPSTREAM_COMMIT,
    SkillRouterEmbeddingRetriever,
    SkillRouterRetriever,
    format_skillrouter_query,
    format_skillrouter_rerank_prompt,
    format_skillrouter_skill,
)


def sample_skill() -> Skill:
    return Skill(
        id="s1",
        name="CSV Filter",
        brief_description="Filter CSV rows.",
        detailed_description="Keep rows matching a column predicate.",
        parameters={
            "type": "object",
            "properties": {"column": {"type": "string"}},
            "required": ["column"],
        },
        returns={"type": "array"},
        tags=["csv"],
    )


def test_official_format_and_provenance() -> None:
    skill = sample_skill()
    query = format_skillrouter_query("filter the csv")
    assert query.endswith("Query:filter the csv")
    document = format_skillrouter_skill(skill)
    assert document.startswith("CSV Filter | Filter CSV rows. | ")
    assert '"parameters"' in document

    prompt = format_skillrouter_rerank_prompt(skill, "filter the csv")
    assert "<Document>: CSV Filter | Filter CSV rows." in prompt
    assert "judge whether the skill document" in prompt

    retriever = SkillRouterEmbeddingRetriever()
    provenance = retriever.provenance()
    assert provenance["encoder_model"] == SKILLROUTER_ENCODER_MODEL
    assert provenance["upstream_commit"] == SKILLROUTER_UPSTREAM_COMMIT
    assert provenance["fallback_allowed"] is False


def test_missing_optional_dependencies_fail_without_fallback() -> None:
    retriever = SkillRouterEmbeddingRetriever(model_name="unused")
    try:
        retriever.index([sample_skill()])
    except Exception:
        assert retriever.backend == "skillrouter-official-open-model"
        assert retriever.provenance()["fallback_allowed"] is False
    else:
        raise AssertionError("invalid published model must not silently succeed")


def test_pipeline_strictly_caps_reranker_candidates() -> None:
    skills = [replace(sample_skill(), id=f"s{index}") for index in range(24)]
    retriever = SkillRouterRetriever(retrieval_top_k=20)
    retriever._skills = skills
    retriever._embeddings = object()
    requested: list[int] = []

    def fake_first_stage(self, query: str, top_k: int = 10) -> RetrievalResult:
        requested.append(top_k)
        return RetrievalResult(
            query=query,
            skills=skills[:top_k],
            scores=[float(-index) for index in range(top_k)],
        )

    retriever._rerank_scores = lambda query, candidates: [
        float(-index) for index in range(len(candidates))
    ]
    with patch.object(SkillRouterEmbeddingRetriever, "retrieve", fake_first_stage):
        result = retriever.retrieve("query", top_k=len(skills))

    assert requested == [20]
    assert len(result.skills) == 20


if __name__ == "__main__":
    test_official_format_and_provenance()
    test_missing_optional_dependencies_fail_without_fallback()
    test_pipeline_strictly_caps_reranker_candidates()
    print("published retriever tests passed")
