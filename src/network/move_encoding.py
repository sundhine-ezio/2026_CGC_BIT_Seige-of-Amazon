"""走法特征编码（走法打分架构）。

把 Amazon 走法 (from, to, arrow) 三元组编码为固定长度特征向量，供
走法打分网络（CandidateMoveNet）的 move_encoder 使用。

Amazon 走法由三个棋盘格 (row, col) 完全确定，无需 draughts 的路径/吃子
维度；6 维特征 = 起点/终点/箭点坐标各自除以 (BOARD_SIZE - 1) 归一化。

坐标系与 board_to_tensor 一致（绝对坐标，不做视角翻转）：棋盘张量已用
己方/对方通道 + 行棋方通道表达视角，走法坐标与之对应。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.game.constants import BOARD_SIZE
from src.game.move import Move

#: 走法特征维度（起点/终点/箭点的 row,col 各 1 维，共 6 维）。
MOVE_FEATURE_DIM: int = 6


def _norm(value: float) -> float:
    """坐标归一化到 [0, 1]。"""
    return value / (BOARD_SIZE - 1)


def encode_move(move: Move) -> np.ndarray:
    """将单个走法编码为 (MOVE_FEATURE_DIM,) 的 float32 特征向量。"""
    fr, fc = move.from_pos
    tr, tc = move.to_pos
    ar, ac = move.arrow_pos
    return np.array(
        [_norm(fr), _norm(fc), _norm(tr), _norm(tc), _norm(ar), _norm(ac)],
        dtype=np.float32,
    )


def encode_legal_moves(moves: Sequence[Move]) -> np.ndarray:
    """将一批走法编码为 (len(moves), MOVE_FEATURE_DIM) 特征矩阵。"""
    if not moves:
        return np.zeros((0, MOVE_FEATURE_DIM), dtype=np.float32)
    return np.stack([encode_move(move) for move in moves], axis=0)


def moves_to_features(indices: np.ndarray) -> np.ndarray:
    """把 (..., 3) 的 [from_idx, to_idx, arrow_idx] 索引转为 (..., 6) 特征。

    与 encode_legal_moves 输出一致（同一走法得到相同特征），供训练时从
    replay buffer 的 (B, K, 3) 索引现算走法特征，避免在样本里冗余存储。

    Args:
        indices: (..., 3) 整数数组，每个元素 ∈ [0, 100)，为 r*10+c 的扁平下标。

    Returns:
        (..., MOVE_FEATURE_DIM) float32 特征数组。
    """
    idx = indices.astype(np.float32)
    row = np.floor(idx / float(BOARD_SIZE))
    col = idx - row * float(BOARD_SIZE)
    features = np.stack(
        [row[..., 0], col[..., 0], row[..., 1], col[..., 1], row[..., 2], col[..., 2]],
        axis=-1,
    )
    return (features / (BOARD_SIZE - 1)).astype(np.float32)
