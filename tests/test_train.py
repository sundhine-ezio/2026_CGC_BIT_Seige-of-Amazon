"""network/train.py 单元测试（训练循环 + checkpoint 保存/加载）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.ai.mcts import decode_move_key, move_key
from src.game.board import Board
from src.game.rules import Rules
from src.network.config import NetworkConfig, TrainingConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import ReplayBuffer, Sample
from src.network.train import load_checkpoint, save_checkpoint, train_cycle


def _small_config() -> NetworkConfig:
    return NetworkConfig(
        num_residual_blocks=1,
        num_filters=8,
        board_feature_dim=8,
        move_embedding_dim=8,
        hidden_dim=8,
    )


def _make_samples(n: int) -> list[Sample]:
    """从初始局面构造 n 个合法样本（K=1）。"""
    board = Board()
    state = AmazonsNet.board_to_tensor(board).astype(np.int8)
    legal = Rules.get_legal_moves(board)
    samples: list[Sample] = []
    for i in range(n):
        moves = np.array([decode_move_key(move_key(legal[i]))], dtype=np.int16)
        pi = np.array([1.0], dtype=np.float32)
        z = 1.0 if i % 2 == 0 else -1.0
        samples.append((state, moves, pi, z))
    return samples


def test_train_cycle_losses() -> None:
    """train_cycle 应返回有限且键完整的损失字典。"""
    model = AmazonsNet(_small_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    buffer = ReplayBuffer(capacity=100)
    for _ in range(10):
        buffer.push(_make_samples(4))
    config = TrainingConfig(batch_size=4, training_steps_per_cycle=2)

    losses = train_cycle(model, optimizer, None, buffer, config)

    assert set(losses.keys()) == {"policy_loss", "value_loss", "total_loss"}
    assert np.isfinite(losses["total_loss"])


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """保存再加载 checkpoint 后参数应一致。"""
    model = AmazonsNet(_small_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(model, optimizer, path, cycle=3)
    loaded = load_checkpoint(path, _small_config())

    for p1, p2 in zip(model.parameters(), loaded.parameters(), strict=False):
        assert torch.equal(p1, p2)
