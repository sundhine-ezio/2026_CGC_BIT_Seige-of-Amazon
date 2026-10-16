"""network/move_encoding.py 单元测试。"""

from __future__ import annotations

import numpy as np

from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules
from src.network.move_encoding import (
    MOVE_FEATURE_DIM,
    encode_legal_moves,
    moves_to_features,
)


def _move_to_indices(move: Move) -> np.ndarray:
    """将走法转成 [from_idx, to_idx, arrow_idx]（与 mcts.decode_move_key 一致）。"""
    fr, fc = move.from_pos
    tr, tc = move.to_pos
    ar, ac = move.arrow_pos
    return np.array([fr * 10 + fc, tr * 10 + tc, ar * 10 + ac], dtype=np.int64)


def test_move_feature_dim() -> None:
    """走法特征维度应为 6。"""
    assert MOVE_FEATURE_DIM == 6


def test_encode_legal_moves_shape_and_range() -> None:
    """编码一批走法应得到 (M, 6) 且在 [0, 1] 内。"""
    moves = Rules.get_legal_moves(Board())
    features = encode_legal_moves(moves)
    assert features.shape == (len(moves), MOVE_FEATURE_DIM)
    assert features.dtype == np.float32
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_moves_to_features_matches_encode() -> None:
    """moves_to_features(索引) 应与 encode_legal_moves(走法) 结果一致。"""
    moves = Rules.get_legal_moves(Board())
    indices = np.stack([_move_to_indices(m) for m in moves], axis=0)
    from_indices = moves_to_features(indices)
    from_moves = encode_legal_moves(moves)
    np.testing.assert_allclose(from_indices, from_moves, atol=1e-6)
