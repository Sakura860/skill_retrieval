"""双塔训练、确定性和 checkpoint 回归；需要 requirements-ml.txt。"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


if TORCH_AVAILABLE:
    import torch

    from core.schemas import Skill, Task
    from data.dataset import RetrievalDataset
    from retrieval.model import RetrievalModel
    from retrieval.train import train


    def _fixture() -> tuple[list[Task], list[Skill]]:
        labels = [
            ("alpha", "orchard"),
            ("beta", "harbor"),
            ("gamma", "forest"),
            ("delta", "desert"),
        ]
        skills = [
            Skill(
                id=f"s_{name}",
                name=f"{name}_tool",
                brief_description=f"handle {topic} records",
                detailed_description=f"execute the {name} operation for {topic} records",
            )
            for name, topic in labels
        ]
        tasks = [
            Task(
                id=f"t_{name}",
                instruction=f"execute the {name} operation for {topic} records",
                expected_skills=[f"s_{name}"],
            )
            for name, topic in labels
        ]
        return tasks, skills


    def _train_once() -> RetrievalModel:
        tasks, skills = _fixture()
        model = RetrievalModel(dim=16, vocab_size=256, seed=37)
        model.index(skills)
        assert model.retrieve(tasks[0].instruction, top_k=1).skills
        train(
            RetrievalDataset(tasks, skills, num_negatives=3),
            model,
            epochs=20,
            batch_size=4,
            lr=0.02,
            temperature=0.05,
            seed=91,
            verbose=False,
        )
        # 训练后旧索引必须失效，防止用训练前的 Skill 向量计算排名。
        assert model.retrieve(tasks[0].instruction, top_k=1).skills == []
        return model


    def test_training_is_deterministic_and_reduces_loss() -> None:
        first = _train_once()
        second = _train_once()
        assert first.training_history == second.training_history
        assert first.training_history[-1] < first.training_history[0]
        for name, tensor in first.net.state_dict().items():
            assert torch.equal(tensor, second.net.state_dict()[name])


    def test_checkpoint_roundtrip_preserves_rankings_and_scores() -> None:
        from main import build_retriever

        tasks, skills = _fixture()
        model = _train_once()
        model.index(skills)
        before = [model.retrieve(task.instruction, top_k=4) for task in tasks]

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dual_encoder.pt"
            model.save_checkpoint(path, metadata={"protocol_id": "unit-test-v01"})
            loaded = RetrievalModel.load_checkpoint(path)
            assert loaded.checkpoint_metadata["protocol_id"] == "unit-test-v01"
            assert loaded.training_history == model.training_history
            assert loaded.retrieve(tasks[0].instruction, top_k=1).skills == []
            loaded.index(skills)
            after = [loaded.retrieve(task.instruction, top_k=4) for task in tasks]
            configured = build_retriever(
                {
                    "retrieval": {
                        "method": "model",
                        "model_checkpoint": str(path),
                    }
                }
            )
            configured.index(skills)
            assert configured.retrieve(tasks[0].instruction, top_k=4).ranked_ids() == (
                after[0].ranked_ids()
            )

        for original, restored in zip(before, after):
            assert original.ranked_ids() == restored.ranked_ids()
            assert len(original.scores) == len(restored.scores)
            assert all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
                for left, right in zip(original.scores, restored.scores)
            )


    def test_checkpoint_rejects_wrong_architecture() -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wrong.pt"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "not-the-dual-encoder",
                    "tokenization_version": "sha256-char-utf8-first128-v1",
                    "config": {},
                    "state_dict": {},
                },
                path,
            )
            try:
                RetrievalModel.load_checkpoint(path)
            except ValueError as exc:
                assert "架构" in str(exc)
            else:
                raise AssertionError("错误架构的 checkpoint 不应被加载")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("dual encoder training tests skipped: install requirements-ml.txt")
    else:
        test_training_is_deterministic_and_reduces_loss()
        test_checkpoint_roundtrip_preserves_rankings_and_scores()
        test_checkpoint_rejects_wrong_architecture()
        print("dual encoder training tests passed")
