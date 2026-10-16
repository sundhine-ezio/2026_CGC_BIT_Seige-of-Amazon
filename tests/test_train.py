"""network/train.py 单元测试（训练循环 + checkpoint 保存/加载）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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


class TestSaveCheckpointAtomic:
    """save_checkpoint 必须原子写：中途失败不能毁掉已有 checkpoint。

    回归背景：原先直接 torch.save 到目标路径，写入途中被杀会留下截断的坏
    文件——对 best_model.pt 而言等于同时失去旧的好模型。长跑（20+ 小时）里
    这种中断是现实风险，所以改为先写 .tmp 再 os.replace。
    """

    def test_saves_and_leaves_no_tmp(self, tmp_path) -> None:
        import torch as _torch

        from src.network.config import NetworkConfig
        from src.network.model import AmazonsNet
        from src.network.train import load_checkpoint, save_checkpoint

        net_config = NetworkConfig(num_residual_blocks=2, num_filters=16)
        model = AmazonsNet(net_config)
        optimizer = _torch.optim.SGD(model.parameters(), lr=0.01)

        target = str(tmp_path / "sub" / "best_model.pt")
        save_checkpoint(model, optimizer, target, cycle=7)

        import os

        assert os.path.isfile(target)
        assert not os.path.exists(f"{target}.tmp"), "不应残留临时文件"
        reloaded = load_checkpoint(target, net_config)
        assert sum(p.numel() for p in reloaded.parameters()) == sum(
            p.numel() for p in model.parameters()
        )

    def test_failure_keeps_old_checkpoint(self, tmp_path, monkeypatch) -> None:
        """写入途中失败时，原有 checkpoint 必须完好无损。"""
        import os

        import torch as _torch

        from src.network.config import NetworkConfig
        from src.network.model import AmazonsNet
        from src.network.train import load_checkpoint, save_checkpoint

        net_config = NetworkConfig(num_residual_blocks=2, num_filters=16)
        model = AmazonsNet(net_config)
        optimizer = _torch.optim.SGD(model.parameters(), lr=0.01)
        target = str(tmp_path / "best_model.pt")

        # 先正常存一份（当作"已有的好模型"）
        save_checkpoint(model, optimizer, target, cycle=1)
        before = os.path.getsize(target)

        # 让下一次 torch.save 在写入途中抛错
        real_save = _torch.save

        def boom(obj, path, *a, **kw):  # type: ignore[no-untyped-def]
            real_save(obj, path, *a, **kw)   # 先写出部分内容
            raise RuntimeError("模拟写入途中被杀")

        monkeypatch.setattr(_torch, "save", boom)
        with pytest.raises(RuntimeError):
            save_checkpoint(model, optimizer, target, cycle=2)

        monkeypatch.undo()
        # 原文件必须仍然可用
        assert os.path.getsize(target) == before
        load_checkpoint(target, net_config)
