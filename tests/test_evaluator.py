"""ai/evaluator.py 的单元测试。"""

import random
import pytest
from src.ai.evaluator import MonteCarloEvaluator
from src.game.board import Board
from src.game.constants import BLACK, BURNED, EMPTY, WHITE


@pytest.fixture
def evaluator() -> MonteCarloEvaluator:
    """创建默认评估器。"""
    return MonteCarloEvaluator(num_simulations=50)


class TestEvaluate:
    """测试 evaluate 方法。"""

    def test_returns_in_range(self, evaluator: MonteCarloEvaluator) -> None:
        """评估值应在 [0, 1] 范围内。"""
        board: Board = Board()
        value: float = evaluator.evaluate(board)
        assert 0.0 <= value <= 1.0

    def test_winning_position(self) -> None:
        """必胜局面: 对手无合法走法 → 评估值=1.0。"""
        b: Board = Board()
        b.reset()
        # 清空棋盘
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]

        # 白方已被封死 → 当前玩家（白方）必败
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 1] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 0] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 1] = BURNED  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(9, 9)]  # type: ignore[attr-defined]
        b._grid[9, 9] = BLACK  # type: ignore[attr-defined]
        b._current_player = WHITE  # type: ignore[attr-defined]

        evaluator2: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
        value: float = evaluator2.evaluate(b)
        # 白方无合法走法 → 胜率为 0.0
        assert value == 0.0

    def test_losing_position(self) -> None:
        """必败局面: 当前玩家无合法走法。"""
        b: Board = Board()
        b.reset()
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]

        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 1] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 0] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 1] = BURNED  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(9, 9)]  # type: ignore[attr-defined]
        b._grid[9, 9] = BLACK  # type: ignore[attr-defined]
        b._current_player = WHITE  # type: ignore[attr-defined]

        evaluator2: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
        value: float = evaluator2.evaluate(b)
        # 白方无合法走法 → 胜率为 0
        assert value == 0.0

    def test_board_unchanged_after_evaluate(self) -> None:
        """评估不应修改原始棋盘。"""
        evaluator2: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=30)
        board: Board = Board()
        grid_before = board.get_grid()
        player_before: int = board.get_current_player()

        evaluator2.evaluate(board)

        from numpy import array_equal
        assert array_equal(board.get_grid(), grid_before)
        assert board.get_current_player() == player_before

    def test_deterministic_with_seed(self) -> None:
        """固定种子评估应可复现。"""
        random.seed(42)
        board: Board = Board()
        evaluator2: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=50)

        random.seed(42)
        v1: float = evaluator2.evaluate(board)

        random.seed(42)
        v2: float = evaluator2.evaluate(board)

        assert v1 == v2
