"""Published retrieval baselines with explicit provenance and no silent fallback.

The SkillRouter implementation follows the open-model inference recipe released
by the authors.  Model loading is lazy so the rest of the project and its mock
tests do not require the heavyweight optional dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.schemas import RetrievalResult, Skill

from .base import BaseRetriever


SKILLROUTER_REPOSITORY = "https://github.com/zhengyanzhao1997/SkillRouter"
SKILLROUTER_UPSTREAM_COMMIT = "2f0c69fe6786bfee6312a1ab5d5f69abdc6bd245"
SKILLROUTER_ENCODER_MODEL = "pipizhao/SkillRouter-Embedding-0.6B"
SKILLROUTER_RERANKER_MODEL = "pipizhao/SkillRouter-Reranker-0.6B"
SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)
SKILLROUTER_RERANK_INSTRUCTION = (
    "Given a coding task description, judge whether the skill document "
    "is relevant and useful for completing the task"
)


def format_skillrouter_query(raw_query: str, max_len: int = 1500) -> str:
    """Use the exact query prefix from the authors' open-model runner."""
    return f"{SKILLROUTER_QUERY_INSTRUCTION}{raw_query[:max_len]}"


def skillrouter_skill_fields(skill: Skill) -> tuple[str, str, str]:
    """Map this project's structured Skill record to name/description/body.

    The mapping is intentionally explicit because it is part of the evaluation
    protocol, rather than an invisible preprocessing choice.
    """
    body = {
        "detailed_description": skill.detailed_description,
        "parameters": skill.parameters,
        "returns": skill.returns,
        "tags": skill.tags,
        "examples": skill.examples,
        "dependencies": skill.dependencies,
        "metadata": skill.metadata,
    }
    return (
        skill.name,
        skill.brief_description,
        json.dumps(body, ensure_ascii=False, sort_keys=True),
    )


def format_skillrouter_skill(
    skill: Skill,
    desc_max: int = 300,
    body_max: int = 2500,
) -> str:
    """Use the authors' ``name | description | body`` document format."""
    name, description, body = skillrouter_skill_fields(skill)
    return f"{name} | {description[:desc_max]} | {body[:body_max]}"


def format_skillrouter_rerank_prompt(
    skill: Skill,
    query_text: str,
    prompt_format: str = "flat-full",
    desc_max: int = 500,
    body_max: int = 2000,
) -> str:
    """Mirror the official flat/structured reranker prompt formats."""
    name, description, body = skillrouter_skill_fields(skill)
    description = description[:desc_max]
    body = body[:body_max]
    if prompt_format == "flat-nd":
        document = f"{name} | {description}"
    elif prompt_format == "flat-full":
        document = f"{name} | {description} | {body}"
    elif prompt_format == "struct":
        return (
            f"<Instruct>: {SKILLROUTER_RERANK_INSTRUCTION}\n\n"
            f"<Query>: {query_text}\n\n"
            f"<Skill>:\n<Name>: {name}\n<Description>: {description}\n"
            f"<Body>: {body}"
        )
    else:
        raise ValueError(f"未知 SkillRouter prompt_format: {prompt_format}")
    return (
        f"<Instruct>: {SKILLROUTER_RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query_text}\n\n<Document>: {document}"
    )


@dataclass(frozen=True)
class SkillRouterProvenance:
    repository: str = SKILLROUTER_REPOSITORY
    upstream_commit: str = SKILLROUTER_UPSTREAM_COMMIT
    encoder_model: str = SKILLROUTER_ENCODER_MODEL
    reranker_model: str | None = None
    retrieval_top_k: int | None = None
    prompt_format: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "SkillRouter",
            "repository": self.repository,
            "upstream_commit": self.upstream_commit,
            "encoder_model": self.encoder_model,
            "reranker_model": self.reranker_model,
            "retrieval_top_k": self.retrieval_top_k,
            "prompt_format": self.prompt_format,
            "fallback_allowed": False,
            "skill_field_mapping": {
                "name": "Skill.name",
                "description": "Skill.brief_description",
                "body": (
                    "JSON(detailed_description, parameters, returns, tags, "
                    "examples, dependencies, metadata)"
                ),
            },
        }


class SkillRouterEmbeddingRetriever(BaseRetriever):
    """Official SkillRouter 0.6B encoder adapted to local Skill records."""

    def __init__(
        self,
        model_name: str = SKILLROUTER_ENCODER_MODEL,
        max_length: int = 4096,
        batch_size: int = 8,
        device: str = "auto",
        dtype: str = "auto",
    ):
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length 和 batch_size 必须为正整数")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.requested_device = device
        self.requested_dtype = dtype
        self.device = "uninitialized"
        self.dtype = "uninitialized"
        self.backend = "skillrouter-official-open-model"
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._skills: list[Skill] = []
        self._embeddings = None

    def provenance(self) -> dict[str, Any]:
        data = SkillRouterProvenance(encoder_model=self.model_name).as_dict()
        data.update({
            "device": self.device,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        })
        return data

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "SkillRouter 官方基线需要可选依赖 torch 和 transformers；"
                "未安装时不会回退到 hash 或其他模型。"
            ) from exc

        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        if self.requested_dtype == "auto":
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            dtype = getattr(torch, self.requested_dtype, None)
            if dtype is None:
                raise ValueError(f"未知 torch dtype: {self.requested_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device).eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.device = str(device)
        self.dtype = str(dtype).replace("torch.", "")

    def prepare(self) -> None:
        """Load the published encoder outside index/query latency windows."""
        self._load_model()

    def _encode(self, texts: list[str]):
        self._load_model()
        torch = self._torch
        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self._tokenizer(
                texts[start:start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                mask = encoded["attention_mask"]
                left_padding = mask[:, -1].sum() == mask.shape[0]
                if left_padding:
                    embeddings = outputs.last_hidden_state[:, -1]
                else:
                    sequence_lengths = mask.sum(dim=1) - 1
                    embeddings = outputs.last_hidden_state[
                        torch.arange(mask.shape[0], device=mask.device),
                        sequence_lengths,
                    ]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu())
        if not all_embeddings:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(all_embeddings, dim=0)

    def index(self, skills: list[Skill]) -> None:
        self._skills = list(skills)
        self._embeddings = self._encode(
            [format_skillrouter_skill(skill, desc_max=500, body_max=8000)
             for skill in self._skills]
        )

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or not self._skills or self._embeddings is None:
            return RetrievalResult(query=query)
        query_embedding = self._encode([
            format_skillrouter_query(query, max_len=2000)
        ])[0]
        similarities = (self._embeddings @ query_embedding).numpy()
        order = np.argsort(-similarities, kind="stable")[:top_k].tolist()
        return RetrievalResult(
            query=query,
            skills=[self._skills[index] for index in order],
            scores=[float(similarities[index]) for index in order],
        )


class SkillRouterRetriever(SkillRouterEmbeddingRetriever):
    """Official encoder top-N followed by the official 0.6B reranker."""

    def __init__(
        self,
        model_name: str = SKILLROUTER_ENCODER_MODEL,
        reranker_model: str = SKILLROUTER_RERANKER_MODEL,
        retrieval_top_k: int = 20,
        max_length: int = 4096,
        reranker_max_length: int = 4096,
        batch_size: int = 8,
        reranker_batch_size: int = 2,
        prompt_format: str = "flat-full",
        device: str = "auto",
        dtype: str = "auto",
    ):
        super().__init__(model_name, max_length, batch_size, device, dtype)
        if retrieval_top_k < 1 or reranker_max_length < 1 or reranker_batch_size < 1:
            raise ValueError("SkillRouter reranker 数值参数必须为正整数")
        if prompt_format not in {"flat-full", "flat-nd", "struct"}:
            raise ValueError(f"未知 SkillRouter prompt_format: {prompt_format}")
        self.reranker_model = reranker_model
        self.retrieval_top_k = retrieval_top_k
        self.reranker_max_length = reranker_max_length
        self.reranker_batch_size = reranker_batch_size
        self.prompt_format = prompt_format
        self.backend = "skillrouter-official-open-pipeline"
        self._reranker = None
        self._reranker_tokenizer = None

    def provenance(self) -> dict[str, Any]:
        data = SkillRouterProvenance(
            encoder_model=self.model_name,
            reranker_model=self.reranker_model,
            retrieval_top_k=self.retrieval_top_k,
            prompt_format=self.prompt_format,
        ).as_dict()
        data.update({
            "device": self.device,
            "dtype": self.dtype,
            "encoder_max_length": self.max_length,
            "reranker_max_length": self.reranker_max_length,
            "encoder_batch_size": self.batch_size,
            "reranker_batch_size": self.reranker_batch_size,
        })
        return data

    def _load_reranker(self) -> None:
        if self._reranker is not None:
            return
        self._load_model()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - guarded by _load_model
            raise RuntimeError("SkillRouter reranker 需要 transformers") from exc
        dtype = getattr(self._torch, self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model,
            padding_side="left",
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.reranker_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(self.device).eval()
        self._reranker_tokenizer = tokenizer
        self._reranker = model

    def prepare(self) -> None:
        """Load both published models before latency measurement starts."""
        self._load_model()
        self._load_reranker()

    def _rerank_scores(self, query: str, candidates: list[Skill]) -> list[float]:
        self._load_reranker()
        tokenizer = self._reranker_tokenizer
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        tokenized = []
        for skill in candidates:
            text = format_skillrouter_rerank_prompt(
                skill, query, prompt_format=self.prompt_format
            )
            encoded = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=(
                    self.reranker_max_length
                    - len(prefix_tokens)
                    - len(suffix_tokens)
                ),
                return_attention_mask=False,
            )
            tokenized.append(prefix_tokens + encoded["input_ids"] + suffix_tokens)

        scores: list[float] = []
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        torch = self._torch
        for start in range(0, len(tokenized), self.reranker_batch_size):
            batch = tokenized[start:start + self.reranker_batch_size]
            max_len = max(len(item) for item in batch)
            padded = [[pad_id] * (max_len - len(item)) + item for item in batch]
            masks = [[0] * (max_len - len(item)) + [1] * len(item) for item in batch]
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self._reranker(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits[:, -1, :]
                values = (logits[:, yes_id] - logits[:, no_id]).float().cpu()
            scores.extend(float(value) for value in values.tolist())
        return scores

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if top_k <= 0 or not self._skills or self._embeddings is None:
            return RetrievalResult(query=query)
        first_stage = super().retrieve(
            query,
            top_k=min(self.retrieval_top_k, len(self._skills)),
        )
        scores = self._rerank_scores(query, first_stage.skills)
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        order = order[:min(top_k, len(order))]
        return RetrievalResult(
            query=query,
            skills=[first_stage.skills[index] for index in order],
            scores=[scores[index] for index in order],
        )
