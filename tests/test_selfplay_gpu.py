"""selfplay_gpu.py 单元测试（GPU 批量 MCTS 自我对弈）。"""

from __future__ import annotations

import numpy as np

from src.game.board import Board
from src.game.rules import Rules
from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample
from src.network.selfplay_gpu import generate_self_play_data_gpu


def _tiny_configs() -> tuple[NetworkConfig, MCTSConfig]:
    """极小配置：1 残差块、8 滤波器、10 次模拟，用于快速验证。"""
    net = NetworkConfig(
        num_residual_blocks=1,
        num_filters=8,
        board_feature_dim=8,
        move_embedding_dim=8,
        hidden_dim=8,
    )
    mcts = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)
    return net, mcts


def _assert_valid_sample(sample: Sample) -> None:
    """校验单个训练样本的结构与 dtype。"""
    state, moves, probs, z = sample
    assert state.shape == (6, 10, 10)
    assert state.dtype == np.int8
    assert moves.ndim == 2 and moves.shape[1] == 3
    assert moves.dtype == np.int16
    assert probs.shape == (moves.shape[0],)
    assert probs.dtype == np.float32
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    assert z in (-1.0, 1.0)


def test_predict_batch_matches_single_predict() -> None:
    """predict_batch 对单局面应与 predict 结果一致。"""
    net_config, _ = _tiny_configs()
    model = AmazonsNet(net_config)
    model.eval()

    board = Board()
    for _ in range(4):
        board.make_move(Rules.get_legal_moves(board)[0])
    moves = Rules.get_legal_moves(board)

    logits_single, value_single = model.predict(board, moves)
    logits_list, values = model.predict_batch([board], [moves])

    assert len(logits_list) == 1
    assert values.shape == (1,)
    np.testing.assert_allclose(logits_list[0], logits_single, atol=1e-5)
    np.testing.assert_allclose(values[0], value_single, atol=1e-5)


def test_gpu_batched_self_play_collects_valid_games() -> None:
    """批量自对弈应产出结构合法、标签正确的整局样本。"""
    net_config, mcts_config = _tiny_configs()
    model = AmazonsNet(net_config)
    model.eval()

    games = generate_self_play_data_gpu(
        model, num_games=2, mcts_config=mcts_config, batch_size=2
    )

    assert len(games) == 2
    for game in games:
        assert len(game) > 0
        for sample in game:
            _assert_valid_sample(sample)
