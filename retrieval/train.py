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
) -> RetrievalModel:
    """联合使用批内负例和显式负例训练。"""
    if len(dataset) == 0:
        raise ValueError("训练数据集为空")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda samples: samples,
    )
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            q = torch.stack([model.net.encode_query(s.query) for s in batch])
            p = torch.stack([model.net.encode_skill(s.pos_skill_text) for s in batch])
            logits = (q @ p.T) / temperature
            labels = torch.arange(len(batch), device=logits.device)
            loss = F.cross_entropy(logits, labels)

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
                    F.cross_entropy(candidate_logits.unsqueeze(0), labels.new_zeros(1))
                )
            if explicit_losses:
                loss = loss + torch.stack(explicit_losses).mean()

            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[train] epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.4f}")
    return model
