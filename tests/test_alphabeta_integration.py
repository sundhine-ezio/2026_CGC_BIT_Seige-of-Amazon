"""Alpha-Beta vs Random 集成测试。"""

from __future__ import annotations

import random

from src.ai.alphabeta import AlphaBetaEngine
from src.ai.evaluator import MonteCarloEvaluator
from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules


def _random_engine(board: Board, time_budget_ms: float) -> Move:  # noqa: ARG001
    """随机选择合法走法（作为基准对手）。"""
    moves: list[Move] = Rules.get_legal_moves(board)
    return random.choice(moves)


def _run_single_game(
    engine_white,  # type: ignore[type-arg]
    engine_black,  # type: ignore[type-arg]
    time_budget_ms: float = 30000.0,
) -> int:
    """运行一局完整对弈，返回胜者（WHITE=1 / BLACK=-1）。"""
    board: Board = Board()
    max_moves: int = 200
    move_count: int = 0

    while not Rules.is_game_over(board) and move_count < max_moves:
        engine = engine_white if board.get_current_player() == 1 else engine_black
        move: Move = engine(board.clone(), time_budget_ms)  # type: ignore[call-arg]
        board.make_move(move)
        move_count += 1

    return Rules.get_winner(board)


def test_alphabeta_beats_random() -> None:
    """AlphaBeta(depth=3) 对 Random 胜率 > 80%（20 局小样本）。"""
    random.seed(42)
    evaluator: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=30)
    ab_engine: AlphaBetaEngine = AlphaBetaEngine(
        evaluator=evaluator, max_depth=3, time_budget_ms=60000.0,
        use_iterative_deepening=False, use_transposition_table=False,
    )

    def ab_search(board: Board, time_ms: float) -> Move:
        return ab_engine.search(board, time_ms)

    ab_wins: int = 0
    random_wins: int = 0
    num_games: int = 20

    for i in range(num_games):
        # 交替先行
        if i % 2 == 0:
            winner: int = _run_single_game(ab_search, _random_engine)
            if winner == 1:
                ab_wins += 1
            else:
                random_wins += 1
        else:
            winner = _run_single_game(_random_engine, ab_search)
            if winner == -1:
                ab_wins += 1
            else:
                random_wins += 1

    win_rate: float = ab_wins / num_games
    print(f"\n[集成测试] AlphaBeta vs Random: {ab_wins}/{num_games} ({win_rate:.0%})")
    assert win_rate > 0.80, f"胜率 {win_rate:.0%} 低于阈值 80%"


def test_alphabeta_vs_alphabeta_completes() -> None:
    """AlphaBeta vs AlphaBeta 能完成完整对弈（3 局）。"""
    evaluator: MonteCarloEvaluator = MonteCarloEvaluator(num_simulations=20)
    engine: AlphaBetaEngine = AlphaBetaEngine(
        evaluator=evaluator, max_depth=2, time_budget_ms=30000.0,
        use_iterative_deepening=False, use_transposition_table=False,
    )

    def search(board: Board, time_ms: float) -> Move:
        return engine.search(board, time_ms)

    for _ in range(3):
        winner = _run_single_game(search, search, time_budget_ms=30000.0)
        assert winner in (1, -1)  # 有明确的胜者
