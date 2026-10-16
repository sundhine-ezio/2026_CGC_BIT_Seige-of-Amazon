"""TrainingConfig.from_dict 的映射测试。

回归背景：这段映射原先内联在 main.py 的 run_train_mode 里，手写字段列表。
新增 `gpu_self_play_devices` / `gpu_self_play_workers` / `gpu_games_per_round`
/ `checkpoint_dir` 时漏了透传，于是配置文件里写的值被**静默忽略**——
`checkpoint_dir` 失效会导致正式训练直接覆盖已有的 models/best_model.pt。

下面的 test_every_field_is_read_from_dict 逐一校验每个字段都能从字典读入，
同类漏字段问题一旦再现就会立刻变红。
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from src.network.config import TrainingConfig


def test_defaults_when_dict_empty() -> None:
    """空字典 → 全部用默认值。"""
    cfg = TrainingConfig.from_dict({})
    assert cfg.batch_size == 512
    assert cfg.self_play_backend == "cpu"
    assert cfg.checkpoint_dir == "models"
    assert cfg.gpu_self_play_workers == 0


def test_every_field_is_read_from_dict() -> None:
    """每个字段都必须能从字典读入——漏一个就说明配置会被静默忽略。"""
    # 按字段类型构造一个明显不同于默认值的样本值
    samples: dict[str, object] = {
        "batch_size": 64,
        "learning_rate": 0.5,
        "momentum": 0.5,
        "weight_decay": 0.5,
        "lr_schedule": "step",
        "training_steps_per_cycle": 7,
        "replay_buffer_capacity": 9,
        "self_play_games_per_cycle": 11,
        "eval_games": 13,
        "eval_num_simulations": 17,
        "eval_win_rate_threshold": 0.9,
        "num_training_cycles": 3,
        "eval_every": 4,
        "checkpoint_every": 6,
        "parallel": True,
        "num_workers": 5,
        "self_play_backend": "gpu_parallel",
        "gpu_batch_size": 16,
        "gpu_self_play_devices": "0,1",
        "gpu_self_play_workers": 32,
        "gpu_games_per_round": 7,
        "checkpoint_dir": "models/run13x256",
    }

    field_names = {f.name for f in fields(TrainingConfig)}
    missing = field_names - set(samples)
    assert not missing, f"测试未覆盖这些字段，请补充样本值: {sorted(missing)}"

    cfg = TrainingConfig.from_dict(dict(samples))

    for name in sorted(field_names):
        actual = getattr(cfg, name)
        expected = samples[name]
        assert actual == expected, (
            f"字段 {name} 未被 from_dict 读入（配置会被静默忽略）："
            f"期望 {expected!r}，实际 {actual!r}"
        )


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("gpu_self_play_devices", "0,1", [0, 1]),
        ("checkpoint_dir", "models/abc", "models/abc"),
    ],
)
def test_gpu_parallel_fields_round_trip(key: str, value: str, expected: object) -> None:
    """gpu_parallel 相关字段与 checkpoint_dir 必须原样透传。"""
    cfg = TrainingConfig.from_dict({key: value})
    actual = getattr(cfg, key)
    if isinstance(expected, list):
        from src.network.selfplay_gpu_pool import parse_devices

        assert parse_devices(actual) == expected
    else:
        assert actual == expected
