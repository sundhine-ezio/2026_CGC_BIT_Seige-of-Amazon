"""网络训练超参数配置（MCTS+ 第二阶段）。

定义网络架构、训练、MCTS 搜索三组超参数，供 model / selfplay / train
与 main.py 复用。全部字段均有保守默认值，便于独立测试与冒烟验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NetworkConfig:
    """网络架构超参数。"""

    board_size: int = 10
    input_channels: int = 6            # 5 核心通道 + 1 可选「最近一步标记」通道
    num_residual_blocks: int = 13
    num_filters: int = 256
    board_feature_dim: int = 256       # 棋盘特征向量维度（打分头/价值头共用输入）
    move_embedding_dim: int = 128      # 走法嵌入维度
    hidden_dim: int = 128              # 打分头 / 价值头隐层维度


@dataclass
class TrainingConfig:
    """训练超参数。"""

    batch_size: int = 512
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"        # 余弦退火
    training_steps_per_cycle: int = 1000
    replay_buffer_capacity: int = 300000
    self_play_games_per_cycle: int = 200
    eval_games: int = 100
    eval_num_simulations: int = 0      # 对抗评估用的 MCTS 模拟次数（0 = 与 self-play 相同）
    eval_win_rate_threshold: float = 0.55
    num_training_cycles: int = 100
    eval_every: int = 10               # 每多少个 cycle 做一次对抗评估
    checkpoint_every: int = 10         # 每多少个 cycle 保存一次 checkpoint
    parallel: bool = False             # 是否用多进程并行自我对弈
    num_workers: int = 0               # 并行 worker 数（0 = 自动取 CPU 核数）
    self_play_backend: str = "cpu"     # self-play 后端：见下方说明
    gpu_batch_size: int = 64           # GPU 批量 self-play 的并发局数（= 每次 GPU 前向 board 数）
    # self_play_backend 取值：
    #   "gpu"          主进程单进程批量 MCTS。GPU 利用率实测仅约 6%，CPU 单核是
    #                  瓶颈，因此大机器上用它等于浪费核数。
    #   "gpu_parallel" 多进程 + GPU：每个 worker 进程各自在指定 GPU 上跑批量
    #                  MCTS，权重按 cycle 广播。这是唯一能同时吃满多核与显卡的
    #                  后端，大机器上应选它。
    #   "cpu"          多进程 CPU 池，网络在 CPU 上推理。
    gpu_self_play_devices: str = "0"   # gpu_parallel 用的 GPU 列表，如 "0,1"（轮转分配）
    gpu_self_play_workers: int = 0     # gpu_parallel 的 worker 进程数（0 = 按显卡数自动取 16×卡数）
    gpu_games_per_round: int = 4       # 每个 worker 每轮跑几局再回传
    checkpoint_dir: str = "models"     # checkpoint 输出目录（不同实验用不同目录，避免互相覆盖）

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """从 config.yaml 的 training 段构造（缺省字段用默认值）。

        集中映射在一处是为了**可测试**：此前这段逻辑内联在 main.py 里，新增
        字段时漏了透传，导致配置文件里写的值被静默忽略——`checkpoint_dir`
        失效会直接覆盖已有的 best_model.pt。tests/test_train_config.py 里有一条
        测试逐一校验"每个字段都能从字典读入"，防止同类问题再现。

        Args:
            data: 配置字典（通常来自 YAML 的 training 段）

        Returns:
            TrainingConfig 实例
        """
        return cls(
            batch_size=int(data.get("batch_size", 512)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            momentum=float(data.get("momentum", 0.9)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            lr_schedule=str(data.get("lr_schedule", "cosine")),
            training_steps_per_cycle=int(data.get("training_steps_per_cycle", 1000)),
            replay_buffer_capacity=int(data.get("replay_buffer_capacity", 300000)),
            self_play_games_per_cycle=int(
                data.get("self_play_games_per_cycle", 200)
            ),
            eval_games=int(data.get("eval_games", 100)),
            eval_num_simulations=int(data.get("eval_num_simulations", 0)),
            eval_win_rate_threshold=float(
                data.get("eval_win_rate_threshold", 0.55)
            ),
            num_training_cycles=int(data.get("num_training_cycles", 100)),
            eval_every=int(data.get("eval_every", 10)),
            checkpoint_every=int(data.get("checkpoint_every", 10)),
            parallel=bool(data.get("parallel", False)),
            num_workers=int(data.get("num_workers", 0)),
            self_play_backend=str(data.get("self_play_backend", "cpu")),
            gpu_batch_size=int(data.get("gpu_batch_size", 64)),
            gpu_self_play_devices=str(data.get("gpu_self_play_devices", "0")),
            gpu_self_play_workers=int(data.get("gpu_self_play_workers", 0)),
            gpu_games_per_round=int(data.get("gpu_games_per_round", 0)),
            checkpoint_dir=str(data.get("checkpoint_dir", "models")),
        )


@dataclass
class MCTSConfig:
    """MCTS 搜索超参数。"""

    num_simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 0.0           # 正式对弈（贪心取访问最多者）
    self_play_temperature: float = 1.0  # 自我对弈前 30 步
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
