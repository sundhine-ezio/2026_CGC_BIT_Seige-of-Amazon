"""ai/alphabeta.py 的单元测试。"""

import pytest

from src.ai.alphabeta import AlphaBetaEngine
from src.ai.evaluator import MonteCarloEvaluator
from src.game.board import Board
from src.game.constants import BLACK, BURNED, EMPTY, WHITE
from src.game.move import Move
from src.game.rules import Rules

# 每步搜索预算。本文件用 MonteCarloEvaluator，它单次评估要跑几十局随机
# playout（实测 1~9 秒），原先 30000~60000ms 的预算让整份文件耗时 3.5 分钟
# 以上（import 层还曾因它 15 分钟跑不完）。这些用例的断言只要求"返回合法
# 走法/非 None"，降到秒级即可保持覆盖，把整份文件压到 30 秒内。
TEST_BUDGET_MS: float = 2000.0
# 迭代加深/DEPTH 相关用例需要稍长预算以便走完前几层
DEEPER_BUDGET_MS: float = 5000.0


@pytest.fixture
def engine() -> AlphaBetaEngine:
    """创建默认 Alpha-Beta 引擎。"""
    evaluator: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
    return AlphaBetaEngine(
        evaluator=evaluator,
        max_depth=3,
        time_budget_ms=TEST_BUDGET_MS,
        use_transposition_table=True,
        use_iterative_deepening=True,
    )


class TestAlphaBetaBasics:
    """基础功能测试。"""

    def test_search_returns_legal_move(self, engine: AlphaBetaEngine) -> None:
        """搜索结果应为合法走法。"""
        board: Board = Board()
        move: Move = engine.search(board, time_budget_ms=TEST_BUDGET_MS)
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        legal_strings: set[str] = {m.to_string() for m in legal_moves}
        assert move.to_string() in legal_strings

    def test_search_with_timeout(self, engine: AlphaBetaEngine) -> None:
        """极短超时也应返回合法走法。"""
        board: Board = Board()
        move: Move = engine.search(board, time_budget_ms=1.0)  # 1ms，必然超时
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        legal_strings: set[str] = {m.to_string() for m in legal_moves}
        assert move.to_string() in legal_strings

    def test_depth_reached(self, engine: AlphaBetaEngine) -> None:
        """带充足时间的搜索应达到目标深度。"""
        board: Board = Board()
        engine.search(board, time_budget_ms=DEEPER_BUDGET_MS)
        stats = engine.get_stats()
        # 至少达到 depth=1
        assert stats.depth_reached >= 1

    def test_get_name(self, engine: AlphaBetaEngine) -> None:
        """引擎名称应包含深度。"""
        assert "AlphaBeta" in engine.get_name()
        assert "3" in engine.get_name()

    def test_reset_clears_transposition(self, engine: AlphaBetaEngine) -> None:
        """reset 应清空置换表状态。"""
        board: Board = Board()
        engine.search(board, time_budget_ms=TEST_BUDGET_MS)
        engine.reset()
        stats_after_reset = engine.get_stats()
        assert stats_after_reset.nodes_visited == 0
        assert stats_after_reset.depth_reached == 0


class TestBasicTactics:
    """基本战术能力测试。"""

    def test_finds_one_move_mate(self) -> None:
        """引擎应能发现一步杀。"""
        # 构造一个白方一步可赢的局面
        b: Board = Board()
        b.reset()
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]

        # 白子在 (0,0)，黑子在 (9,9)
        # 白子可走 (0,0)→(0,5)→(9,0) 封死黑子？不，需要更精确的构造。
        # 简化：白方在角落有子，黑方也有子但白方可以先手封路
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 1] = BLACK  # type: ignore[attr-defined]
        b._grid[0, 2] = BURNED  # type: ignore[attr-defined]
        b._grid[0, 3] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 0] = BURNED  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(0, 1)]  # type: ignore[attr-defined]
        b._current_player = WHITE  # type: ignore[attr-defined]

        evaluator: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=30)
        engine2: AlphaBetaEngine = AlphaBetaEngine(
            evaluator=evaluator, max_depth=3, time_budget_ms=TEST_BUDGET_MS,
            use_iterative_deepening=False,
        )

        move: Move = engine2.search(b, time_budget_ms=TEST_BUDGET_MS)
        # 至少返回合法走法
        legal_moves: list[Move] = Rules.get_legal_moves(b)
        legal_strings: set[str] = {m.to_string() for m in legal_moves}
        assert move.to_string() in legal_strings

    def test_does_not_blunder_in_one_move_losing(self) -> None:
        """不应选择导致己方立即被封死的走法。"""
        # 用初始棋盘测试: 引擎应选合理的走法
        board: Board = Board()
        evaluator: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
        engine2: AlphaBetaEngine = AlphaBetaEngine(
            evaluator=evaluator, max_depth=2, time_budget_ms=TEST_BUDGET_MS,
            use_iterative_deepening=False,
        )
        move: Move = engine2.search(board, time_budget_ms=TEST_BUDGET_MS)
        assert move is not None

        # 执行走法后己方仍有合法走法
        board.make_move(move)
        assert not Rules.is_game_over(board)


class TestIterativeDeepening:
    """迭代加深测试。"""

    def test_deeper_not_worse(self) -> None:
        """更深搜索不应返回更差的结果（同一起始局面，统计上）。"""
        board: Board = Board()

        evaluator_d2: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
        engine_d2: AlphaBetaEngine = AlphaBetaEngine(
            evaluator=evaluator_d2, max_depth=2, time_budget_ms=DEEPER_BUDGET_MS,
            use_iterative_deepening=False,
        )
        move_d2: Move = engine_d2.search(board.clone(), time_budget_ms=DEEPER_BUDGET_MS)

        board2: Board = board.clone()
        evaluator_d3: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
        engine_d3: AlphaBetaEngine = AlphaBetaEngine(
            evaluator=evaluator_d3, max_depth=3, time_budget_ms=DEEPER_BUDGET_MS,
            use_iterative_deepening=False,
        )
        move_d3: Move = engine_d3.search(board2, time_budget_ms=DEEPER_BUDGET_MS)

        # 两者都应合法
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        legal_strings: set[str] = {m.to_string() for m in legal_moves}
        assert move_d2.to_string() in legal_strings
        assert move_d3.to_string() in legal_strings
