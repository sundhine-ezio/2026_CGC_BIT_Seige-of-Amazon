"""network/selfplay.py 单元测试（自我对弈数据生成 + 对抗评估）。"""

from __future__ import annotations

from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.selfplay import generate_self_play_data, pit_evaluate


def _small_model() -> AmazonsNet:
    model = AmazonsNet(
        NetworkConfig(
            num_residual_blocks=1,
            num_filters=8,
            board_feature_dim=8,
            move_embedding_dim=8,
            hidden_dim=8,
        )
    )
    model.eval()
    return model


def _small_mcts_config() -> MCTSConfig:
    return MCTSConfig(
        num_simulations=2,
        c_puct=1.0,
        temperature=0.0,
        self_play_temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,  # 关闭噪声，保证确定性
    )


def test_generate_self_play_data() -> None:
    """自我对弈应生成结构正确的 (state, moves, pi, z) 样本。"""
    model = _small_model()
    games = generate_self_play_data(model, num_games=1, mcts_config=_small_mcts_config())
    assert len(games) == 1
    samples = games[0]
    assert len(samples) > 0
    for state, moves, pi, z in samples:
        assert state.shape == (6, 10, 10)
        assert moves.ndim == 2 and moves.shape[1] == 3
        assert pi.shape == (moves.shape[0],)
        assert abs(float(pi.sum()) - 1.0) < 1e-5
        assert z in (-1.0, 1.0)


def test_pit_evaluate() -> None:
    """对抗评估应返回结构正确且自洽的统计结果。"""
    result = pit_evaluate(
        _small_model(), _small_model(), num_games=2, mcts_config=_small_mcts_config()
    )
    assert set(result.keys()) == {"new_win", "old_win", "new_win_rate"}
    assert result["new_win"] + result["old_win"] == 2
    assert 0.0 <= result["new_win_rate"] <= 1.0
