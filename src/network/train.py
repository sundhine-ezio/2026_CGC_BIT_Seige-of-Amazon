"""训练流水线（MCTS+ 第二阶段）。

实现单轮训练（条件三头策略损失 + 价值损失）与完整「自我对弈 → 训练 →
对抗评估 → 替换」闭环，以及 checkpoint 的保存/加载。
"""

from __future__ import annotations

import copy
import os
import signal
import sys
from dataclasses import replace
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  # torch 惯用别名

from src.network.config import MCTSConfig, NetworkConfig, TrainingConfig
from src.network.model import AmazonsNet
from src.network.move_encoding import moves_to_features
from src.network.replay_buffer import ReplayBuffer, Sample
from src.network.selfplay import generate_self_play_data, pit_evaluate
from src.network.selfplay_gpu import generate_self_play_data_gpu
from src.network.selfplay_gpu_pool import GpuSelfPlayPool, parse_devices
from src.network.selfplay_parallel import ParallelSelfPlayPool


class _SelfPlayPool(Protocol):
    """self-play 工作池的公共接口（CPU 池与 GPU 池都满足）。"""

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        """广播最新权重（CPU state_dict）。"""
        ...

    def collect_games(self, num_games: int) -> list[list[Sample]]:
        """收集整局样本。"""
        ...

    def shutdown(self) -> None:
        """停止所有 worker。"""
        ...


class _Scheduler(Protocol):
    """学习率调度器协议（解耦 torch.optim.lr_scheduler 的类型）。"""

    def step(self) -> None:
        """执行一次学习率调度步进。"""
        ...


def _resolve_device(device: str) -> torch.device:
    """解析设备字符串，CUDA 不可用时回退 CPU。"""
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退 CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device)


def _cpu_state_dict(model: AmazonsNet) -> dict[str, torch.Tensor]:
    """返回模型的 CPU state_dict（用于广播给并行 self-play worker）。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _terminate_gracefully(signum: int, frame: Any) -> None:
    """将 SIGTERM 转为 SystemExit，触发 train_loop 的 finally 清理 worker。

    默认情况下 SIGTERM（kill 命令）会直接终止进程而不执行 finally，导致
    daemon worker 被孤儿化残留；转成 SystemExit 后 finally 会正常执行
    pool.shutdown()。
    """
    sys.exit(0)


def train_cycle(
    model: AmazonsNet,
    optimizer: torch.optim.Optimizer,
    scheduler: _Scheduler | None,
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
) -> dict[str, float]:
    """单轮训练：采样 → 前向 → 计算 L_total → 反向 → 更新。

    Args:
        model: 待训练网络
        optimizer: 优化器
        scheduler: 学习率调度器（可为 None）
        replay_buffer: 经验回放池
        config: 训练超参数

    Returns:
        {"policy_loss": float, "value_loss": float, "total_loss": float}
    """
    model.train()
    device = next(model.parameters()).device

    total_policy = 0.0
    total_value = 0.0

    for _ in range(config.training_steps_per_cycle):
        batch = replay_buffer.sample(config.batch_size)
        state = torch.from_numpy(batch["state"]).to(device)
        pi = torch.from_numpy(batch["pi"]).to(device)
        z = torch.from_numpy(batch["z"]).to(device)

        moves_np = batch["moves"]
        move_features = torch.from_numpy(moves_to_features(moves_np)).to(device)
        move_mask = torch.from_numpy(
            (moves_np.sum(axis=-1) > 0).astype(np.float32)
        ).to(device)

        out = model(state, move_features, move_mask)
        policy_loss = _policy_loss(out["policy_logits"], pi, move_mask)
        value_loss = F.mse_loss(out["value"], z)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch stub 未标注 backward
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_policy += float(policy_loss.item())
        total_value += float(value_loss.item())

    n = config.training_steps_per_cycle
    return {
        "policy_loss": total_policy / n,
        "value_loss": total_value / n,
        "total_loss": (total_policy + total_value) / n,
    }


def _policy_loss(
    policy_logits: torch.Tensor,
    pi: torch.Tensor,
    move_mask: torch.Tensor,
) -> torch.Tensor:
    """走法打分策略 soft 交叉熵损失。

    L_policy = -Σ_k π_k · log softmax(logits)_k，补零（非法）位置贡献为 0。

    Args:
        policy_logits: (B, K) 每个候选走法的打分 logits
        pi:            (B, K) 访问次数概率（soft 目标）
        move_mask:     (B, K) 合法走法掩码（1 合法 / 0 补零）
    """
    log_probs = F.log_softmax(policy_logits, dim=1)  # (B, K)
    safe_log_probs = torch.where(
        move_mask.bool(), log_probs, torch.zeros_like(log_probs)
    )
    per_sample = -(pi * safe_log_probs).sum(dim=1)  # (B,)
    return per_sample.mean()


def train_loop(
    model: AmazonsNet,
    config: TrainingConfig,
    mcts_config: MCTSConfig,
    device: str = "cuda",
) -> None:
    """完整训练闭环（自我对弈 → 训练 → 对抗评估 → 替换）。

    Args:
        model: 待训练网络（会在内部移到目标设备）
        config: 训练超参数
        mcts_config: MCTS 搜索超参数
        device: 目标设备字符串（"cuda" 或 "cpu"）
    """
    torch_device = _resolve_device(device)

    # SIGTERM（kill）→ SystemExit，触发下方 finally 里的 pool.shutdown()，
    # 避免 worker 进程被孤儿化残留占用 CPU。
    signal.signal(signal.SIGTERM, _terminate_gracefully)

    # 日志重定向到文件时 stdout 默认是块缓冲，会导致 tail -f 看不到实时进度；
    # 显式改为行缓冲，使每个 cycle 的 print 立即落盘。
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    # self-play 后端选择：
    #   "gpu_parallel" 多进程 + GPU（多核大机器首选；唯一能同时吃满多核与显卡的）
    #   "gpu"          单进程批量 MCTS（实测 GPU 利用率仅约 6%，CPU 单核是瓶颈）
    #   "cpu"          多进程 CPU 池，网络在 CPU 上推理
    backend = config.self_play_backend
    use_gpu_selfplay = backend == "gpu"

    pool: _SelfPlayPool | None = None
    # GPU 池单独留一个强类型引用：评估要调它的并行 evaluate（CPU 池没有这个方法）
    gpu_pool: GpuSelfPlayPool | None = None
    if backend == "gpu_parallel":
        devices = parse_devices(config.gpu_self_play_devices)
        if config.gpu_self_play_workers > 0:
            num_workers = config.gpu_self_play_workers
        else:
            # 实测约 16 个单进程才能喂饱一张卡（GPU 利用率 6.4% × 16 ≈ 100%）
            num_workers = 16 * len(devices)
        # 每个 worker 每轮的局数。显式配置优先；否则按每 cycle 目标局数摊分并
        # 向上取整，保证一轮产量（= worker 数 × 每轮局数）≥ 目标局数。
        if config.gpu_games_per_round > 0:
            games_per_round = config.gpu_games_per_round
        else:
            games_per_round = max(
                1, -(-config.self_play_games_per_cycle // num_workers)
            )
        gpu_pool = GpuSelfPlayPool(
            num_workers,
            model.config,
            mcts_config,
            devices,
            batch_size=config.gpu_batch_size,
            games_per_round=games_per_round,
        )
        pool = gpu_pool
        # 首次广播：触发第一轮自对弈（cycle 1 的 collect_games 等它）
        pool.broadcast_weights(_cpu_state_dict(model))
        print(
            f"[train] GPU 多进程 self-play: {num_workers} workers / "
            f"GPU {devices} / 每 worker batch {config.gpu_batch_size} / "
            f"每轮 {gpu_pool.round_games} 局",
            flush=True,
        )
    elif config.parallel and not use_gpu_selfplay:
        # 多进程 CPU 池。必须在 model.to(cuda) 之前创建：fork 发生在 CUDA
        # 初始化前，避免子进程继承 CUDA 上下文导致死锁/崩溃。
        num_workers = (
            config.num_workers if config.num_workers > 0 else (os.cpu_count() or 1)
        )
        pool = ParallelSelfPlayPool(num_workers, model.config, mcts_config)
        pool.broadcast_weights(_cpu_state_dict(model))
        print(f"[train] 并行 self-play: {num_workers} workers", flush=True)
    elif use_gpu_selfplay:
        print(
            f"[train] GPU 批量 self-play (batch_size={config.gpu_batch_size})",
            flush=True,
        )

    model.to(torch_device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    total_steps = config.num_training_cycles * config.training_steps_per_cycle
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)
    best_model = copy.deepcopy(model)
    best_win_rate = 0.0
    checkpoint_dir = config.checkpoint_dir

    try:
        for cycle in range(1, config.num_training_cycles + 1):
            print(
                f"[train] cycle {cycle}/{config.num_training_cycles}: self-play "
                f"({config.self_play_games_per_cycle} games)..."
            )
            if use_gpu_selfplay:
                games = generate_self_play_data_gpu(
                    model,
                    config.self_play_games_per_cycle,
                    mcts_config,
                    config.gpu_batch_size,
                )
            elif pool is not None:
                games = pool.collect_games(config.self_play_games_per_cycle)
            else:
                games = generate_self_play_data(
                    model, config.self_play_games_per_cycle, mcts_config
                )
            for game in games:
                replay_buffer.push(game)
            print(f"[train] buffer size = {len(replay_buffer)}, training...")

            losses = train_cycle(model, optimizer, scheduler, replay_buffer, config)
            print(
                f"[train] policy_loss={losses['policy_loss']:.4f} "
                f"value_loss={losses['value_loss']:.4f} "
                f"total_loss={losses['total_loss']:.4f}"
            )

            if cycle % config.eval_every == 0:
                print(f"[train] pit evaluation ({config.eval_games} games)...")
                # 评估口径：单独调低模拟次数，用更便宜的方式换更勤的进度观测。
                eval_sims = config.eval_num_simulations or mcts_config.num_simulations
                if config.eval_num_simulations > 0:
                    print(
                        f"[train] （评估用 {config.eval_num_simulations} 模拟）",
                        flush=True,
                    )
                if gpu_pool is not None:
                    # 并行走 worker 池：同样的对局在 GPU 上批量推进，
                    # 单局成本约降到 1/30，而且一轮能拿到更多局（统计更可靠）
                    result = gpu_pool.evaluate(
                        _cpu_state_dict(model),
                        _cpu_state_dict(best_model),
                        config.eval_games,
                        sims=eval_sims,
                    )
                else:
                    eval_mcts_config = mcts_config
                    if config.eval_num_simulations > 0:
                        eval_mcts_config = replace(
                            mcts_config, num_simulations=config.eval_num_simulations
                        )
                    result = pit_evaluate(
                        model, best_model, config.eval_games, eval_mcts_config
                    )
                print(
                    f"[train] new_win_rate={result['new_win_rate']:.3f} "
                    f"({int(result.get('games', config.eval_games))} 局, "
                    f"threshold {config.eval_win_rate_threshold})"
                )
                if result["new_win_rate"] > config.eval_win_rate_threshold:
                    best_model = copy.deepcopy(model)
                    best_win_rate = result["new_win_rate"]
                    save_checkpoint(
                        model, optimizer,
                        os.path.join(checkpoint_dir, "best_model.pt"), cycle,
                    )
                    print(f"[train] new best model (win_rate={best_win_rate:.3f})")

            # 广播放到评估之后：评估要靠 worker 池执行，先广播会让 worker
            # 忙着跑自对弈、评估拿不到结果。
            if pool is not None:
                pool.broadcast_weights(_cpu_state_dict(model))

            if cycle % config.checkpoint_every == 0:
                path = os.path.join(checkpoint_dir, f"checkpoint_cycle{cycle}.pt")
                save_checkpoint(model, optimizer, path, cycle)
    finally:
        if pool is not None:
            pool.shutdown()


def save_checkpoint(
    model: AmazonsNet, optimizer: torch.optim.Optimizer, path: str, cycle: int
) -> None:
    """保存模型 + 优化器 + 轮次到 .pt（先写临时文件，再原子重命名）。

    直接 torch.save 到目标路径时，若进程在写入途中被杀（kill -9 / 崩溃 /
    断电），会留下一个被截断的损坏文件——对 best_model.pt 而言，等于同时
    丢掉了旧的那份好模型。改为先写 ``<path>.tmp`` 再 ``os.replace``：重命名
    在同一文件系统内是原子操作，中途失败也能保住原文件完好。

    Args:
        model: 模型
        optimizer: 优化器
        path: 目标路径
        cycle: 当前轮次
    """
    directory: str = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path: str = f"{path}.tmp"
    torch.save(
        {
            "cycle": cycle,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path: str, config: NetworkConfig) -> AmazonsNet:
    """从 .pt 加载模型参数，返回可推理的 AmazonsNet（在 CPU 上）。"""
    model = AmazonsNet(config)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
