"""检索模型训练样本。"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import Skill, Task


@dataclass
class RetrievalSample:
    query: str
    pos_skill_text: str
    neg_skill_texts: list[str]
    task_id: str = ""
    pos_skill_id: str = ""


class RetrievalDataset:
    """根据 expected_skills 构造正负样本。"""

    def __init__(self, tasks: list[Task], skills: list[Skill], num_negatives: int = 4):
        self.samples: list[RetrievalSample] = []
        skill_by_id = {s.id: s for s in skills}
        all_skills = list(skills)

        for task in tasks:
            pos = [skill_by_id[sid] for sid in task.expected_skills if sid in skill_by_id]
            if not pos:
                continue
            positive_ids = set(task.expected_skills)
            for p in pos:
                negs = [s for s in all_skills if s.id not in positive_ids][:num_negatives]
                self.samples.append(RetrievalSample(
                    query=task.instruction,
                    pos_skill_text=p.to_text("detailed"),
                    neg_skill_texts=[n.to_text("detailed") for n in negs],
                    task_id=task.id,
                    pos_skill_id=p.id,
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RetrievalSample:
        return self.samples[idx]
