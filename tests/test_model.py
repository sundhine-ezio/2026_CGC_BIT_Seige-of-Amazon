"""network/model.py 单元测试（AmazonsNet 走法打分架构）。"""

from __future__ import annotations

import numpy as np
import torch

from src.game.board import Board
from src.game.constants import BLACK
from src.game.rules import Rules
from src.network.config import NetworkConfig
from src.network.model import AmazonsNet
from src.network.move_encoding import MOVE_FEATURE_DIM


def _small_config() -> NetworkConfig:
    return NetworkConfig(
        num_residual_blocks=1,
        num_filters=8,
        board_feature_dim=8,
        move_embedding_dim=8,
        hidden_dim=8,
    )


def test_board_to_tensor_shape_and_channels() -> None:
    """board_to_tensor 应输出 (6,10,10)，各通道语义正确。"""
    board = Board()
    tensor = AmazonsNet.board_to_tensor(board)
    assert tensor.shape == (6, 10, 10)
    assert int(tensor[0].sum()) == 4  # 己方（白）亚马逊
    assert int(tensor[1].sum()) == 4  # 对方（黑）亚马逊
    assert int(tensor[4].sum()) == 100  # 白方行棋 → 全 1
    assert set(np.unique(tensor[0])) <= {0.0, 1.0}


def test_board_to_tensor_black_indicator() -> None:
    """黑方行棋时通道 4 应全为 0。"""
    board = Board()
    board._current_player = BLACK
    tensor = AmazonsNet.board_to_tensor(board)
    assert int(tensor[4].sum()) == 0


def test_forward_shapes() -> None:
    """前向输出各头的形状与取值域应正确。"""
    model = AmazonsNet(_small_config())
    model.eval()
    x = torch.randn(2, 6, 10, 10)
    move_features = torch.randn(2, 5, MOVE_FEATURE_DIM)
    with torch.no_grad():
        out = model(x, move_features)
    assert out["policy_logits"].shape == (2, 5)
    assert out["value"].shape == (2, 1)
    assert float(out["value"].min()) >= -1.0
    assert float(out["value"].max()) <= 1.0


def test_predict_returns_logits_and_value() -> None:
    """predict 应返回与合法走法数等长的 logits 与 [-1,1] 范围内的价值。"""
    model = AmazonsNet(_small_config())
    model.eval()
    board = Board()
    moves = Rules.get_legal_moves(board)
    logits, value = model.predict(board, moves)
    assert logits.shape == (len(moves),)
    assert logits.dtype == np.float32
    assert -1.0 <= value <= 1.0
