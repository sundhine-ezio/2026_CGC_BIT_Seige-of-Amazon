"""多进程 GPU 并行自我对弈池（同时承担并行对抗评估）。

为什么需要它：`selfplay_gpu` 的批量 MCTS 是**单进程**的，而自对弈的瓶颈是
CPU 单核的走法生成（实测 GPU 利用率仅约 6%）。所以在多核机器上单进程等于
浪费核数。本模块让 N 个 worker 进程各自在指定 GPU 上跑批量 MCTS，从而同时
吃满多核与显卡。

架构：

  - 进程用 **spawn** 而非 fork 创建。fork 出来的子进程会继承父进程的 CUDA
    上下文，极易死锁/崩溃；spawn 让子进程干净地初始化 CUDA，因此本池的创建
    顺序不受 `model.to(cuda)` 影响。
  - worker 持有**自己那张 GPU 上**的模型；主进程把权重**写成文件**并在共享字典
    里发布「路径 + 版本号」，worker 发现版本更新就从文件读入。为什么不直接把
    state_dict 放进 Manager 字典：那样 torch 的每个张量存储都要复制一个文件
    描述符，13块/256 的 state_dict 有 176 个张量，几十个 worker 同时取就会撞上
    fd 上限（实测报 ``OSError: [Errno 24] Too many open files``）。走文件后每个
    worker 只占一个普通读 fd。
  - **按轮工作**：一轮 = 所有 worker 各跑固定的局数后统一回传。这样一轮的产量
    固定为 num_workers × games_per_round，不会像持续投喂那样在主进程队列里
    积压出用旧策略生成的陈旧样本。
  - worker 支持两种任务，由共享字典的 ``job`` 切换：
      "selfplay"  自对弈产样本（默认）
      "eval"      A/B 对抗评估（双模型，只回传胜负）

关于评估为什么要做成并行的：`selfplay.pit_evaluate` 是串行的，实测 13块/256
模型跑 40 局约 40 分钟，是整个闭环里最贵的一环。放进本池后同样的对局在
worker 上并行推进，单局成本约降到 1/30，而且一轮能拿到更多局数（统计更可靠）。

用法::

    pool = GpuSelfPlayPool(32, net_cfg, mcts_cfg, devices=[0],
                           batch_size=8, games_per_round=7)
    pool.broadcast_weights(cpu_state_dict)     # 每 cycle 一次 → 触发一轮自对弈
    games = pool.collect_games(200)
    stats = pool.evaluate(cur_weights, best_weights, num_games=64, sims=400)
    pool.shutdown()
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import os
import queue as queue_mod
import resource
import shutil
import tempfile
import time
from typing import Any

import torch

from src.game.constants import WHITE
from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample

# worker 等主进程广播权重时的轮询间隔（秒）
_IDLE_POLL_S: float = 0.05
# collect_games / evaluate 等待单个结果的超时；超时后检查 worker 是否还活着
_GET_TIMEOUT_S: float = 30.0
# 估算每个 worker 同时从 Manager 取权重所需的 fd 数（torch 每个张量存储占一个）
_FDS_PER_WORKER: int = 200


def _raise_fd_limit(num_workers: int) -> tuple[int, int]:
    """把本进程的 fd 软上限提到硬上限，返回 (软上限, 硬上限)。

    为什么必须做：worker 从 Manager 取权重时，torch 的每个张量存储都要经
    resource_sharer 复制一个文件描述符。13块/256 的 state_dict 有上百个张量，
    几十个 worker 同时取就会撞上默认的 1024 上限，报
    ``OSError: [Errno 24] Too many open files``。软上限在硬上限之内可以自由
    提高，spawn 出的子进程会继承。

    Args:
        num_workers: worker 数（用于估算需求并给出告警）

    Returns:
        (提升后的软上限, 硬上限)
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            soft = hard
        except (ValueError, OSError):
            pass
    need = num_workers * _FDS_PER_WORKER
    if soft < need:
        print(
            f"[warn] 文件描述符上限 {soft} 可能不足以支撑 {num_workers} 个 worker"
            f"（估算需要 {need}）。请先在 shell 里执行 "
            f"`ulimit -n {need * 2}`，或减少 worker 数。",
            flush=True,
        )
    return soft, hard


def parse_devices(spec: str | list[int]) -> list[int]:
    """把 "0,1" 或 [0, 1] 解析成设备号列表。

    Args:
        spec: 形如 "0,1" 的字符串，或已是列表

    Returns:
        设备号列表（至少一个元素）

    Raises:
        ValueError: 解析不出任何设备号
    """
    if isinstance(spec, list):
        devices = [int(d) for d in spec]
    else:
        devices = [int(p) for p in spec.replace(" ", "").split(",") if p != ""]
    if not devices:
        raise ValueError(f"未解析出任何 GPU 设备号: {spec!r}")
    return devices


def _gpu_worker_loop(
    shared: Any,
    sample_queue: Any,
    eval_queue: Any,
    stop_event: Any,
    network_config: NetworkConfig,
    mcts_config: MCTSConfig,
    worker_index: int,
    device_index: int,
    batch_size: int,
    games_per_round: int,
) -> None:
    """worker 主循环：等本轮权重 → 按 job 执行 → 回传结果。"""
    import torch

    from src.network.selfplay_gpu import generate_self_play_data_gpu

    # 核数很多时必须限制线程数，否则 N 个进程 × 多线程会严重超订
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{device_index}")
    model = AmazonsNet(network_config).to(device)
    model.eval()
    opponent: AmazonsNet | None = None  # 评估用的对手模型，懒创建后复用
    last_version = -1

    while not stop_event.is_set():
        version = int(shared["version"])
        if version == 0 or version == last_version:
            time.sleep(_IDLE_POLL_S)
            continue

        job = str(shared.get("job", "selfplay"))
        path_a = shared.get("self_path" if job != "eval" else "a_path")
        path_b = shared.get("b_path") if job == "eval" else None
        if not path_a or (job == "eval" and not path_b):
            # 权重文件尚未发布：本轮不认领（last_version 不更新，下轮重试）
            time.sleep(_IDLE_POLL_S)
            continue
        if not os.path.isfile(str(path_a)) or (
            path_b is not None and not os.path.isfile(str(path_b))
        ):
            time.sleep(_IDLE_POLL_S)
            continue

        model.load_state_dict(torch.load(str(path_a), map_location="cpu"))
        model.eval()
        last_version = version

        if job == "eval":
            from dataclasses import replace

            from src.network.selfplay_gpu import play_eval_games_gpu

            if opponent is None:
                opponent = AmazonsNet(network_config).to(device)
            if path_b is None:  # 收窄类型（上面已判定非 None）
                continue
            opponent.load_state_dict(torch.load(str(path_b), map_location="cpu"))
            opponent.eval()

            sims = int(shared.get("eval_sims", 0))
            eval_cfg = (
                replace(mcts_config, num_simulations=sims) if sims > 0 else mcts_config
            )
            # 评估局数由 evaluate() 指定（= ceil(num_games / worker 数)）。
            # 不能用自对弈的 games_per_round，否则各 worker 会多跑，主进程只收
            # 前一部分结果，等于只统计了最快的那些 worker，样本有偏。
            eval_games = int(shared.get("eval_games_per_worker", 0)) or games_per_round
            # 各 worker 交错先后手，保证全局黑白平衡
            results = play_eval_games_gpu(
                model,
                opponent,
                eval_games,
                eval_cfg,
                batch_size=batch_size,
                opening_plies=int(shared.get("eval_opening_plies", 8)),
                first_game_a_is_white=(worker_index % 2 == 0),
            )
            for item in results:
                eval_queue.put(item)
        else:
            games = generate_self_play_data_gpu(
                model, games_per_round, mcts_config, batch_size
            )
            for game in games:
                sample_queue.put(game)


class GpuSelfPlayPool:
    """多进程 GPU self-play 工作池（广播权重 + 按轮收集 + 并行评估）。"""

    def __init__(
        self,
        num_workers: int,
        network_config: NetworkConfig,
        mcts_config: MCTSConfig,
        devices: str | list[int],
        batch_size: int = 16,
        games_per_round: int = 4,
        weights_dir: str = "",
    ) -> None:
        """创建并启动 num_workers 个 GPU worker。

        Args:
            num_workers: worker 进程数（须 > 0）
            network_config: 网络架构（worker 据此在各自 GPU 上建模型）
            mcts_config: MCTS 搜索超参数（self-play 用）
            devices: GPU 设备号，如 "0,1"；worker 轮转分配到各卡
            batch_size: 每个 worker 内并发推进的局数
            games_per_round: 每个 worker 每轮跑几局（自对弈一轮产量 = 两者相乘）
            weights_dir: 权重中转目录（空 = 自动建临时目录，退出时清理）

        Raises:
            ValueError: 参数非法
        """
        if num_workers <= 0:
            raise ValueError("num_workers 必须 > 0")
        if batch_size <= 0 or games_per_round <= 0:
            raise ValueError("batch_size 与 games_per_round 必须 > 0")

        device_list = parse_devices(devices)
        # spawn：子进程干净地初始化 CUDA，避免 fork 继承 CUDA 上下文导致死锁
        ctx = mp.get_context("spawn")

        self._manager = ctx.Manager()
        self._shared: Any = self._manager.dict()
        self._shared["version"] = 0
        self._shared["self_path"] = ""
        self._shared["a_path"] = ""
        self._shared["b_path"] = ""
        self._shared["job"] = "selfplay"
        self._shared["eval_sims"] = 0
        self._shared["eval_opening_plies"] = 8
        self._shared["eval_games_per_worker"] = 0
        self._sample_queue: Any = ctx.Queue()
        self._eval_queue: Any = ctx.Queue()
        self._stop_event: Any = ctx.Event()

        # 权重经文件分发（避免 Manager 传输 tensor 复制大量 fd）
        self._weights_dir = weights_dir or tempfile.mkdtemp(prefix="amazons_pool_")
        os.makedirs(self._weights_dir, exist_ok=True)
        self._published: dict[str, list[str]] = {}

        self._num_workers = num_workers
        self._games_per_round = games_per_round
        # 权重经 Manager 传输时每个张量占一个 fd，worker 一多就会撞上限
        soft_fd, _ = _raise_fd_limit(num_workers)
        self._fd_soft_limit = soft_fd
        self._round_games = num_workers * games_per_round
        self._devices = device_list

        self._workers: list[Any] = []
        for i in range(num_workers):
            proc = ctx.Process(
                target=_gpu_worker_loop,
                args=(
                    self._shared,
                    self._sample_queue,
                    self._eval_queue,
                    self._stop_event,
                    network_config,
                    mcts_config,
                    i,
                    device_list[i % len(device_list)],
                    batch_size,
                    games_per_round,
                ),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)

    @property
    def num_workers(self) -> int:
        """worker 进程数。"""
        return self._num_workers

    @property
    def fd_soft_limit(self) -> int:
        """本进程提升后的文件描述符软上限（供调用方记录日志）。"""
        return self._fd_soft_limit

    @property
    def round_games(self) -> int:
        """自对弈一轮的样本局数（= worker 数 × 每 worker 局数）。"""
        return self._round_games

    def _check_workers_alive(self, got: int, want: int, what: str) -> None:
        """有 worker 已退出就抛错，否则主进程会永久阻塞。"""
        dead = [i for i, w in enumerate(self._workers) if not w.is_alive()]
        if dead:
            raise RuntimeError(
                f"self-play worker 已退出（编号 {dead}），"
                f"{what} 只收到 {got}/{want}，训练中止"
            )

    def _publish(self, state_dict: dict[str, Any], tag: str, version: int) -> str:
        """把权重原子写入文件并登记，返回路径。

        先写 .tmp 再 rename：worker 可能随时来读，绝不能让它读到半个文件。

        Args:
            state_dict: CPU state_dict
            tag: 文件名标签（self / a / b）
            version: 版本号

        Returns:
            权重文件路径
        """
        path = os.path.join(self._weights_dir, f"{tag}_v{version}.pt")
        tmp = f"{path}.tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, path)

        # 只保留最近 3 个版本，避免长跑把磁盘塞满
        kept = self._published.setdefault(tag, [])
        kept.append(path)
        while len(kept) > 3:
            stale = kept.pop(0)
            with contextlib.suppress(OSError):
                os.remove(stale)
        return path

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        """发布最新权重（写文件 + 递增版本号），触发下一轮自对弈。"""
        version = int(self._shared["version"]) + 1
        path = self._publish(state_dict, "self", version)
        self._shared["job"] = "selfplay"
        self._shared["self_path"] = path
        self._shared["version"] = version

    def collect_games(self, num_games: int = 0) -> list[list[Sample]]:
        """阻塞收集本轮全部自对弈样本。

        注意：本池按轮工作，一轮产量固定为 round_games，因此**实际返回的局数
        可能多于 num_games**（num_games 仅用于日志提示）。这是刻意的——多收
        的那部分是本轮刚生成的样本，丢弃它们才是浪费。

        Args:
            num_games: 期望局数（仅用于提示，不影响返回条数）

        Returns:
            整局样本列表，长度为 round_games

        Raises:
            RuntimeError: 有 worker 已退出
        """
        games: list[list[Sample]] = []
        while len(games) < self._round_games:
            try:
                games.append(self._sample_queue.get(timeout=_GET_TIMEOUT_S))
            except queue_mod.Empty:
                self._check_workers_alive(len(games), self._round_games, "本轮自对弈")
        return games

    def evaluate(
        self,
        weights_a: dict[str, Any],
        weights_b: dict[str, Any],
        num_games: int,
        sims: int = 0,
        opening_plies: int = 8,
    ) -> dict[str, float]:
        """并行 A/B 对抗评估（A 对 B，各执一半白/黑，开局随机化）。

        Args:
            weights_a: A 方权重（CPU state_dict）
            weights_b: B 方权重（CPU state_dict）
            num_games: 期望对局数；实际局数 = worker 数 × 每 worker 局数
                （向上取整，故通常略多于 num_games）
            sims: 评估用 MCTS 模拟次数（0 = 与 self-play 相同）
            opening_plies: 开局随机化步数

        Returns:
            {"new_win": A 胜局数, "old_win": B 胜局数,
             "new_win_rate": A 胜率, "games": 实际局数}

        Raises:
            ValueError: num_games <= 0
            RuntimeError: 有 worker 已退出
        """
        if num_games <= 0:
            raise ValueError("num_games 必须 > 0")

        per_worker = max(1, math.ceil(num_games / self._num_workers))
        want = self._num_workers * per_worker

        # 清空上一轮可能的残留，避免把旧结果算进来
        while True:
            try:
                self._eval_queue.get_nowait()
            except queue_mod.Empty:
                break

        version = int(self._shared["version"]) + 1
        path_a = self._publish(weights_a, "a", version)
        path_b = self._publish(weights_b, "b", version)
        self._shared["job"] = "eval"
        self._shared["a_path"] = path_a
        self._shared["b_path"] = path_b
        self._shared["eval_sims"] = int(sims)
        self._shared["eval_opening_plies"] = int(opening_plies)
        self._shared["eval_games_per_worker"] = per_worker
        self._shared["version"] = version

        results: list[tuple[int, bool]] = []
        while len(results) < want:
            try:
                results.append(self._eval_queue.get(timeout=_GET_TIMEOUT_S))
            except queue_mod.Empty:
                self._check_workers_alive(len(results), want, "本轮评估")

        a_wins = sum(
            1
            for winner, a_is_white in results
            if (winner == WHITE) == a_is_white
        )
        return {
            "new_win": float(a_wins),
            "old_win": float(len(results) - a_wins),
            "new_win_rate": a_wins / len(results),
            "games": float(len(results)),
        }

    def shutdown(self) -> None:
        """停止所有 worker 并回收资源（含权重中转目录）。"""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        self._manager.shutdown()
        shutil.rmtree(self._weights_dir, ignore_errors=True)
