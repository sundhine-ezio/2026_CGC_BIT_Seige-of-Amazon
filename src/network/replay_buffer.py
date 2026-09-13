"""经验回放池。

存储自我对弈生成的训练样本，支持 FIFO 淘汰与随机 mini-batch 采样。
样本格式见 Sample 类型别名。
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

# 单个训练样本:
#   state: (input_channels, 10, 10) int8，当前行棋方视角
#   moves: (K, 3) int16，K 个「被访问过」的合法走法 (from_idx, to_idx, arrow_idx)
#   pi:    (K,) float32，与 moves 对齐的访问次数概率（联合分布，已归一化）
#   z:     float ∈ {-1, +1}，本样本行棋方最终胜负
Sample = tuple[np.ndarray, np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """经验回放池，存储自我对弈生成的训练样本。"""

    def __init__(self, capacity: int = 300000) -> None:
        """初始化回放池。

        Args:
            capacity: 最大样本数，超过后按 FIFO 丢弃最旧样本。
        """
        self._capacity: int = capacity
        self._buffer: deque[Sample] = deque(maxlen=capacity)

    def push(self, game_data: list[Sample]) -> None:
        """存入一整局棋谱的所有样本（超出容量按 FIFO 淘汰）。"""
        self._buffer.extend(game_data)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """随机采样一个 mini-batch。

        Args:
            batch_size: 采样样本数

        Returns:
            {
                "state": (B, C, 10, 10) float32,
                "moves": (B, K_max, 3) int64,
                "pi":    (B, K_max) float32,
                "z":     (B, 1) float32,
            }
            其中 K_max 为 batch 内最大 K，K 较小的样本以 0 补齐。

        Raises:
            ValueError: 回放池为空
        """
        n: int = len(self._buffer)
        if n == 0:
            raise ValueError("ReplayBuffer is empty")

        if batch_size > n:
            indices: list[int] = [random.randrange(n) for _ in range(batch_size)]
        else:
            indices = random.sample(range(n), batch_size)

        samples: list[Sample] = [self._buffer[i] for i in indices]

        max_k: int = max(len(s[1]) for s in samples)

        states: np.ndarray = np.stack(
            [s[0].astype(np.float32) for s in samples]
        )  # (B, C, 10, 10)
        moves: np.ndarray = np.zeros((batch_size, max_k, 3), dtype=np.int64)
        pi: np.ndarray = np.zeros((batch_size, max_k), dtype=np.float32)
        z: np.ndarray = np.array([[s[3]] for s in samples], dtype=np.float32)

        for b, s in enumerate(samples):
            k: int = len(s[1])
            moves[b, :k, :] = s[1]
            pi[b, :k] = s[2]

        return {"state": states, "moves": moves, "pi": pi, "z": z}

    def __len__(self) -> int:
        """当前样本总数。"""
        return len(self._buffer)
