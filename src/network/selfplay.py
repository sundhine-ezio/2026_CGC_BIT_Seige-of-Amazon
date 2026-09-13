"""自我对弈数据生成与对抗评估（MCTS+ 第二阶段）。

使用当前网络进行自我对弈，生成 (state, moves, pi, z) 训练样本；
并提供新网络 vs 旧网络的对抗评估（pit）。
"""

from __future__ import annotations

import numpy as np

from src.ai.mcts import MCTSEngine, decode_move_key
from src.game.board import Board
from src.game.constants import BLACK, WHITE
from src.game.move import Move
from src.game.rules import Rules
from src.network.config import MCTSConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample


def generate_self_play_data(
    model: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
) -> list[list[Sample]]:
    """使用当前网络自我对弈，生成训练数据。

    每局流程:
      1. 新建棋盘与 MCTSEngine
      2. 循环至终局: 前 30 步 temperature=self_play_temperature 且注入
         Dirichlet 噪声，之后 temperature=0（贪心）；记录每个局面的
         (state, moves, pi, 行棋方)
      3. 终局后按胜负标注 z（行棋方 == 胜者 → +1，否则 -1）

    Args:
        model: 当前网络
        num_games: 对弈局数
        mcts_config: MCTS 搜索超参数

    Returns:
        每局一个样本列表（Sample 定义见 replay_buffer.py）
    """
    return [_self_play_one_game(model, mcts_config) for _ in range(num_games)]


def _self_play_one_game(model: AmazonsNet, mcts_config: MCTSConfig) -> list[Sample]:
    engine = MCTSEngine(
        model,
        num_simulations=mcts_config.num_simulations,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=mcts_config.dirichlet_epsilon,
    )
    board = Board()
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    move_index = 0

    while not Rules.is_game_over(board):
        temperature = mcts_config.self_play_temperature if move_index < 30 else 0.0
        add_noise = move_index < 30 and mcts_config.dirichlet_epsilon > 0.0
        pi, move = engine.get_policy(board, temperature, add_noise=add_noise)

        keys = list(pi.keys())
        moves = np.array([decode_move_key(k) for k in keys], dtype=np.int16)
        probs = np.array([pi[k] for k in keys], dtype=np.float32)
        state = AmazonsNet.board_to_tensor(board).astype(np.int8)

        records.append((state, moves, probs, board.get_current_player()))
        board.make_move(move)
        move_index += 1

    winner = Rules.get_winner(board)
    samples: list[Sample] = []
    for state, moves, probs, player in records:
        z = 1.0 if player == winner else -1.0
        samples.append((state, moves, probs, z))
    return samples


def pit_evaluate(
    new_model: AmazonsNet,
    old_model: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
) -> dict[str, float]:
    """新网络 vs 旧网络对抗评估（temperature=0，各执一半白/黑）。

    Args:
        new_model: 待评估的新网络
        old_model: 基线旧网络
        num_games: 对弈局数
        mcts_config: MCTS 搜索超参数

    Returns:
        {"new_win": N, "old_win": N, "new_win_rate": float}
    """
    new_wins = 0
    for g in range(num_games):
        new_is_white = g % 2 == 0
        white_engine = MCTSEngine(
            new_model if new_is_white else old_model,
            num_simulations=mcts_config.num_simulations,
            c_puct=mcts_config.c_puct,
        )
        black_engine = MCTSEngine(
            old_model if new_is_white else new_model,
            num_simulations=mcts_config.num_simulations,
            c_puct=mcts_config.c_puct,
        )
        winner = _play_game(white_engine, black_engine)
        if (winner == WHITE and new_is_white) or (winner == BLACK and not new_is_white):
            new_wins += 1

    return {
        "new_win": new_wins,
        "old_win": num_games - new_wins,
        "new_win_rate": new_wins / num_games,
    }


def _play_game(white_engine: MCTSEngine, black_engine: MCTSEngine) -> int:
    """用两个引擎对弈一局，返回胜者（WHITE 或 BLACK）。"""
    board = Board()
    while not Rules.is_game_over(board):
        engine = white_engine if board.get_current_player() == WHITE else black_engine
        move: Move = engine.search(board.clone(), 0.0)
        board.make_move(move)
    return Rules.get_winner(board)
