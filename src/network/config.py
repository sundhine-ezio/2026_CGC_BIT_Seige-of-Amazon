"""网络训练超参数配置（MCTS+ 第二阶段）。

定义网络架构、训练、MCTS 搜索三组超参数，供 model / selfplay / train
与 main.py 复用。全部字段均有保守默认值，便于独立测试与冒烟验证。
"""

from __future__ import annotations

from dataclasses import dataclass


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
    eval_win_rate_threshold: float = 0.55
    num_training_cycles: int = 100
    eval_every: int = 10               # 每多少个 cycle 做一次对抗评估
    checkpoint_every: int = 10         # 每多少个 cycle 保存一次 checkpoint
    parallel: bool = False             # 是否用多进程并行自我对弈
    num_workers: int = 0               # 并行 worker 数（0 = 自动取 CPU 核数）
    self_play_backend: str = "cpu"     # self-play 后端："gpu"（批量 MCTS）/ "cpu"（多进程池）
    gpu_batch_size: int = 64           # GPU 批量 self-play 的并发局数（= 每次 GPU 前向 board 数）


@dataclass
class MCTSConfig:
    """MCTS 搜索超参数。"""

    num_simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 0.0           # 正式对弈（贪心取访问最多者）
    self_play_temperature: float = 1.0  # 自我对弈前 30 步
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
