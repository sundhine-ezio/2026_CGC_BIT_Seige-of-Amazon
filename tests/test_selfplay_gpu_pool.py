"""selfplay_gpu_pool 的集成测试（需要 CUDA）。

验证三件事：
  1. 一轮的产量恰好是 num_workers × games_per_round（按轮回传，不积压陈旧样本）
  2. 样本格式与单进程后端一致
  3. 再次广播权重后能继续产出（版本同步不会卡死）
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.selfplay_gpu_pool import GpuSelfPlayPool, parse_devices

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 CUDA 才能测 GPU self-play 池"
)


class TestParseDevices:
    """设备号解析（不需要显卡）。"""

    def test_comma_separated(self) -> None:
        assert parse_devices("0,1") == [0, 1]

    def test_spaces_and_single(self) -> None:
        assert parse_devices(" 0 ") == [0]

    def test_list_passthrough(self) -> None:
        assert parse_devices([1, 0]) == [1, 0]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_devices("")


@requires_cuda
def test_pool_round_count_and_weight_sync() -> None:
    """一轮产量固定、样本格式正确、二次广播后能继续产出。"""
    # 用小模型 + 极少模拟，让测试跑得快
    net_config = NetworkConfig(num_residual_blocks=2, num_filters=32)
    mcts_config = MCTSConfig(
        num_simulations=8,
        c_puct=1.0,
        temperature=0.0,
        self_play_temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    num_workers = 2
    games_per_round = 1
    pool = GpuSelfPlayPool(
        num_workers,
        net_config,
        mcts_config,
        devices=[0],
        batch_size=1,
        games_per_round=games_per_round,
    )
    try:
        assert pool.round_games == num_workers * games_per_round

        model = AmazonsNet(net_config)
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pool.broadcast_weights(weights)

        games = pool.collect_games(0)
        assert len(games) == pool.round_games, "一轮产量应等于 worker 数 × 每轮局数"
        assert all(len(g) > 0 for g in games), "每局都应有样本"

        state, moves, probs, z = games[0][0]
        assert state.shape == (6, 10, 10)
        assert state.dtype == np.int8
        assert moves.ndim == 2 and moves.shape[1] == 3
        assert moves.dtype == np.int16
        assert probs.shape[0] == moves.shape[0]
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-4)
        assert z in (-1.0, 1.0)

        # 再广播一轮：验证 worker 能重新同步权重并继续产出（不会卡在旧版本）
        pool.broadcast_weights(
            {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        games2 = pool.collect_games(0)
        assert len(games2) == pool.round_games
    finally:
        pool.shutdown()


@requires_cuda
def test_pool_rejects_bad_args() -> None:
    """参数非法应立刻报错，而不是起一堆没用的进程。"""
    net_config = NetworkConfig(num_residual_blocks=2, num_filters=32)
    mcts_config = MCTSConfig(num_simulations=4)
    with pytest.raises(ValueError):
        GpuSelfPlayPool(0, net_config, mcts_config, devices=[0])
    with pytest.raises(ValueError):
        GpuSelfPlayPool(1, net_config, mcts_config, devices=[0], batch_size=0)


@requires_cuda
def test_pool_parallel_evaluate_and_switch_back() -> None:
    """并行评估可用，且评估后能切回自对弈（最易卡死的集成点）。

    评估走的是同一个 worker 池，靠共享字典的 job 字段切换任务；若切换逻辑
    有误，主进程会卡在一个永远收不满的队列上——所以这里必须验证"评估 →
    切回自对弈"整条路径。
    """
    net_config = NetworkConfig(num_residual_blocks=2, num_filters=32)
    mcts_config = MCTSConfig(num_simulations=8, c_puct=1.0, temperature=0.0)

    pool = GpuSelfPlayPool(
        2, net_config, mcts_config, devices=[0], batch_size=1, games_per_round=1
    )
    try:
        model = AmazonsNet(net_config)
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # 先跑一轮自对弈，拿到"当前模型"
        pool.broadcast_weights(weights)
        games = pool.collect_games(0)
        assert len(games) == pool.round_games

        # 同一模型自战：各执一半黑白 + 开局随机化 → 胜率应接近 0.5，不应是 0/1
        stats = pool.evaluate(
            weights_a=weights, weights_b=weights, num_games=4, sims=8
        )
        assert set(stats) >= {"new_win", "old_win", "new_win_rate", "games"}
        # 4 局 → 2 workers × ceil(4/2)=2 → 实际 4 局
        assert stats["games"] == 4.0
        assert stats["new_win"] + stats["old_win"] == stats["games"]
        assert 0.0 <= stats["new_win_rate"] <= 1.0

        # 关键：评估之后必须能切回自对弈并继续拿到样本
        pool.broadcast_weights(weights)
        games2 = pool.collect_games(0)
        assert len(games2) == pool.round_games
        assert all(len(g) > 0 for g in games2)
    finally:
        pool.shutdown()


@requires_cuda
def test_pool_evaluate_rejects_zero_games() -> None:
    """num_games <= 0 应立刻报错，而不是干等。"""
    net_config = NetworkConfig(num_residual_blocks=2, num_filters=16)
    pool = GpuSelfPlayPool(
        1, net_config, MCTSConfig(num_simulations=4), devices=[0],
        batch_size=1, games_per_round=1,
    )
    try:
        model = AmazonsNet(net_config)
        w = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with pytest.raises(ValueError):
            pool.evaluate(weights_a=w, weights_b=w, num_games=0)
    finally:
        pool.shutdown()
