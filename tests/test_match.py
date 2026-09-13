"""批量对弈测试工具。"""

from __future__ import annotations

from typing import Callable

from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules


def run_match(
    engine_white: Callable[[Board, float], Move],
    engine_black: Callable[[Board, float], Move],
    num_games: int = 10,
    time_budget_ms: float = 30000.0,
) -> dict[str, float]:
    """运行 num_games 局对弈，返回统计数据。

    Args:
        engine_white: 白方搜索函数 f(board, time_ms) -> Move
        engine_black: 黑方搜索函数 f(board, time_ms) -> Move
        num_games: 局数
        time_budget_ms: 单步最大时间

    Returns:
        {"white_wins": N, "black_wins": N, "avg_moves": float}
    """
    white_wins: int = 0
    black_wins: int = 0
    total_moves: int = 0

    for game_idx in range(num_games):
        board: Board = Board()
        move_count: int = 0
        max_moves: int = 200

        while not Rules.is_game_over(board) and move_count < max_moves:
            engine = engine_white if board.get_current_player() == 1 else engine_black
            move: Move = engine(board.clone(), time_budget_ms)
            board.make_move(move)
            move_count += 1

        total_moves += move_count
        winner: int = Rules.get_winner(board)
        if winner == 1:
            white_wins += 1
        else:
            black_wins += 1

        print(f"  Game {game_idx + 1}/{num_games}: "
              f"{'白胜' if winner == 1 else '黑胜'} ({move_count} 步)")

    avg_moves: float = total_moves / num_games
    print(f"\n结果: 白 {white_wins} : 黑 {black_wins}, "
          f"平均 {avg_moves:.0f} 步/局")

    return {
        "white_wins": float(white_wins),
        "black_wins": float(black_wins),
        "avg_moves": avg_moves,
    }
