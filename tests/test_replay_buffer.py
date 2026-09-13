"""network/replay_buffer.py 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.network.replay_buffer import ReplayBuffer, Sample


def _make_sample(k: int) -> Sample:
    state = np.zeros((6, 10, 10), dtype=np.int8)
    moves = np.zeros((k, 3), dtype=np.int16)
    pi = np.full(k, 1.0 / k, dtype=np.float32)
    return (state, moves, pi, 1.0)


def test_push_and_len() -> None:
    """push 应增加样本数量。"""
    buffer = ReplayBuffer(capacity=10)
    buffer.push([_make_sample(3) for _ in range(5)])
    assert len(buffer) == 5


def test_sample_shapes() -> None:
    """sample 应按 K_max 对齐输出张量形状。"""
    buffer = ReplayBuffer(capacity=10)
    buffer.push([_make_sample(2), _make_sample(4), _make_sample(3)])
    batch = buffer.sample(3)
    assert batch["state"].shape == (3, 6, 10, 10)
    assert batch["moves"].shape == (3, 4, 3)  # K_max = 4
    assert batch["pi"].shape == (3, 4)
    assert batch["z"].shape == (3, 1)


def test_fifo_eviction() -> None:
    """超过容量应按 FIFO 淘汰最旧样本。"""
    buffer = ReplayBuffer(capacity=3)
    for _ in range(5):
        buffer.push([_make_sample(1)])
    assert len(buffer) == 3


def test_sample_empty_raises() -> None:
    """空回放池采样应抛出 ValueError。"""
    buffer = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buffer.sample(1)
