"""network/selfplay_parallel.py 单元测试（多进程并行 self-play）。"""

from __future__ import annotations

import numpy as np

from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample
from src.network.selfplay_parallel import ParallelSelfPlayPool


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


def test_parallel_self_play_collects_valid_games() -> None:
    """并行池应能广播权重并收集到结构合法的整局样本。"""
    net_config, mcts_config = _tiny_configs()
    model = AmazonsNet(net_config)
    model.eval()

    pool = ParallelSelfPlayPool(
        num_workers=2, network_config=net_config, mcts_config=mcts_config
    )
    try:
        pool.broadcast_weights(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        games = pool.collect_games(2)
    finally:
        pool.shutdown()

    assert len(games) == 2
    for game in games:
        assert len(game) > 0
        for sample in game:
            _assert_valid_sample(sample)


def test_pool_rejects_nonpositive_workers() -> None:
    """num_workers <= 0 应抛出 ValueError。"""
    net_config, mcts_config = _tiny_configs()
    try:
        ParallelSelfPlayPool(
            num_workers=0, network_config=net_config, mcts_config=mcts_config
        )
        raise AssertionError("应抛出 ValueError")
    except ValueError:
        pass
