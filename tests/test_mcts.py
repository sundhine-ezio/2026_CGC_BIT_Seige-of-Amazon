"""ai/mcts.py 单元测试（使用假模型，无 PyTorch 依赖）。

覆盖 move_key 编码、MCTSEngine / PureMCTSEngine 走法合法性、
输入棋盘不被修改、策略分布归一化、终局抛错。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.mcts import MCTSEngine, PureMCTSEngine, decode_move_key, move_key
from src.game.board import Board
from src.game.constants import BLACK, BURNED, WHITE
from src.game.move import Move
from src.game.rules import Rules


class _FakeModel:
    """实现 MCTS 所需 predict 接口的最小假网络（均匀先验 + 0 价值）。"""

    def predict(self, board: Board, moves: list[Move]) -> tuple[np.ndarray, float]:
        return np.zeros(len(moves), dtype=np.float32), 0.0


def _terminal_board() -> Board:
    """构造白方唯一亚马逊被完全围死的终局局面。"""
    b = Board()
    b._grid.fill(BURNED)
    b._grid[4, 4] = WHITE
    b._amazon_positions = {WHITE: [(4, 4)], BLACK: []}
    b._current_player = WHITE
    return b


def test_move_key_roundtrip() -> None:
    """move_key 与 decode_move_key 应互为逆运算。"""
    move = Move((3, 4), (0, 0), (9, 9))
    assert decode_move_key(move_key(move)) == (34, 0, 99)


def test_mcts_search_returns_legal_move() -> None:
    """MCTSEngine.search 应返回合法走法，且不改动输入棋盘。"""
    engine = MCTSEngine(_FakeModel(), num_simulations=10, c_puct=1.0)
    board = Board()
    before = board.get_grid()

    move = engine.search(board.clone(), 0.0)

    assert move in Rules.get_legal_moves(board)
    assert np.array_equal(board.get_grid(), before)


def test_mcts_get_policy_normalized() -> None:
    """get_policy 返回的策略分布应归一化，且键均为合法走法。"""
    engine = MCTSEngine(_FakeModel(), num_simulations=20)
    board = Board()
    pi, move = engine.get_policy(board, temperature=1.0)

    assert abs(sum(pi.values()) - 1.0) < 1e-6
    assert move in Rules.get_legal_moves(board)
    legal_keys = {move_key(m) for m in Rules.get_legal_moves(board)}
    assert set(pi.keys()) <= legal_keys


def test_pure_mcts_search_returns_legal_move() -> None:
    """PureMCTSEngine.search 应返回合法走法。"""
    engine = PureMCTSEngine(num_simulations=20)
    board = Board()
    move = engine.search(board.clone(), 0.0)
    assert move in Rules.get_legal_moves(board)


def test_mcts_search_on_terminal_raises() -> None:
    """终局局面调用 MCTSEngine.search 应抛出 ValueError。"""
    engine = MCTSEngine(_FakeModel(), num_simulations=10)
    with pytest.raises(ValueError):
        engine.search(_terminal_board(), 0.0)


def test_pure_mcts_search_on_terminal_raises() -> None:
    """终局局面调用 PureMCTSEngine.search 应抛出 ValueError。"""
    engine = PureMCTSEngine(num_simulations=10)
    with pytest.raises(ValueError):
        engine.search(_terminal_board(), 0.0)
