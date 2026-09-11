"""双塔模型的对比学习训练。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import RetrievalDataset
from retrieval.model import RetrievalModel


def train(
    dataset: RetrievalDataset,
    model: RetrievalModel,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    temperature: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> RetrievalModel:
    """联合使用批内负例和显式负例训练。"""
    if len(dataset) == 0:
        raise ValueError("训练数据集为空")
    if epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if lr <= 0:
        raise ValueError("lr 必须大于 0")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA 训练，但当前 torch.cuda.is_available() 为 False")

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda samples: samples,
    )
    model.invalidate_index()
    model.net.to(resolved_device)
    model.net.train()
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            q = torch.stack([model.net.encode_query(s.query) for s in batch])
            p = torch.stack([model.net.encode_skill(s.pos_skill_text) for s in batch])
            logits = (q @ p.T) / temperature
            # 同一 batch 可包含同一个 Skill 的多条正样本，不能把它们互相当作负例。
            positive_keys = [sample.pos_skill_id or sample.pos_skill_text for sample in batch]
            positive_mask = torch.tensor(
                [
                    [left == right for right in positive_keys]
                    for left in positive_keys
                ],
                dtype=torch.bool,
                device=logits.device,
            )
            loss = (
                torch.logsumexp(logits, dim=1)
                - torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
            ).mean()

            explicit_losses = []
            for query_vec, sample in zip(q, batch):
                if not sample.neg_skill_texts:
                    continue
                negatives = torch.stack([
                    model.net.encode_skill(text) for text in sample.neg_skill_texts
                ])
                positive = model.net.encode_skill(sample.pos_skill_text).unsqueeze(0)
                candidates = torch.cat([positive, negatives], dim=0)
                candidate_logits = (candidates @ query_vec) / temperature
                explicit_losses.append(
                    F.cross_entropy(
                        candidate_logits.unsqueeze(0),
                        torch.zeros(1, dtype=torch.long, device=logits.device),
                    )
                )
            if explicit_losses:
                loss = loss + torch.stack(explicit_losses).mean()

            loss.backward()
            opt.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(loader)
        history.append(epoch_loss)
        if verbose:
            print(f"[train] epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
    model.net.eval()
    model.training_history = history
    model.invalidate_index()
    return model
