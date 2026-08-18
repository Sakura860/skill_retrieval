"""执行轨迹反思。"""
from __future__ import annotations

from core.llm import LLM


class Reflection:
    def __init__(self, llm: LLM):
        self.llm = llm

    def reflect(self, task: str, trajectory: str, success: bool) -> str:
        """总结本次执行并给出下一次改进建议。"""
        outcome = "成功" if success else "失败"
        prompt = (
            f"任务：{task}\n"
            f"执行结果：{outcome}\n"
            f"执行轨迹：\n{trajectory}\n\n"
            "简要反思技能选择、顺序和参数；给出一条可执行的改进建议。"
        )
        return self.llm.generate([{"role": "user", "content": prompt}])
