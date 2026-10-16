"""Alpha-Beta vs Random 集成测试。

关于测试参数：本文件原先用 MonteCarloEvaluator(20~30) + 每步 30 秒预算 +
20 局完整对弈，粗算需要 9 小时以上，实际上永远跑不完（历史文档里它一直标
"运行中"、从未产出结果）。改用 HeuristicEvaluator（单次评估 0.1ms，比
MonteCarlo 快约一万倍）并把每步预算降到 0.4~1 秒后，整份文件可在数分钟内
跑完，断言保持不变。
"""

from __future__ import annotations

import random

from src.ai.alphabeta import AlphaBetaEngine
from src.ai.evaluator import HeuristicEvaluator
from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules

# 每步搜索预算：小到能在测试里跑完，大到足以让迭代加深走完前几层
MOVE_BUDGET_MS: float = 400.0


def _random_engine(board: Board, time_budget_ms: float) -> Move:  # noqa: ARG001
    """随机选择合法走法（作为基准对手）。"""
    moves: list[Move] = Rules.get_legal_moves(board)
    return random.choice(moves)


def _run_single_game(
    engine_white,  # type: ignore[type-arg]
    engine_black,  # type: ignore[type-arg]
    time_budget_ms: float = MOVE_BUDGET_MS,
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
    """AlphaBeta(depth=3) 对 Random 胜率 > 80%（10 局小样本）。"""
    random.seed(42)
    evaluator: HeuristicEvaluator = HeuristicEvaluator()
    ab_engine: AlphaBetaEngine = AlphaBetaEngine(
        evaluator=evaluator, max_depth=3, time_budget_ms=60000.0,
        use_iterative_deepening=True, use_transposition_table=True,
    )

    def ab_search(board: Board, time_ms: float) -> Move:
        return ab_engine.search(board, time_ms)

    ab_wins: int = 0
    num_games: int = 10

    for i in range(num_games):
        # 交替先行
        if i % 2 == 0:
            winner: int = _run_single_game(ab_search, _random_engine, MOVE_BUDGET_MS)
            if winner == 1:
                ab_wins += 1
        else:
            winner = _run_single_game(_random_engine, ab_search, MOVE_BUDGET_MS)
            if winner == -1:
                ab_wins += 1

    win_rate: float = ab_wins / num_games
    print(f"\n[集成测试] AlphaBeta vs Random: {ab_wins}/{num_games} ({win_rate:.0%})")
    assert win_rate > 0.80, f"胜率 {win_rate:.0%} 低于阈值 80%"


def test_alphabeta_vs_alphabeta_completes() -> None:
    """AlphaBeta vs AlphaBeta 能完成完整对弈（2 局）。"""
    evaluator: HeuristicEvaluator = HeuristicEvaluator()
    engine: AlphaBetaEngine = AlphaBetaEngine(
        evaluator=evaluator, max_depth=2, time_budget_ms=60000.0,
        use_iterative_deepening=True, use_transposition_table=True,
    )

    def search(board: Board, time_ms: float) -> Move:
        return engine.search(board, time_ms)

    for _ in range(2):
        winner = _run_single_game(search, search, time_budget_ms=MOVE_BUDGET_MS)
        assert winner in (1, -1)  # 有明确的胜者
