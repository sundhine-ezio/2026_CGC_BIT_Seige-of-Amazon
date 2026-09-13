"""多进程并行自我对弈（AlphaZero 风格）。

用多个 worker 进程并行跑 self-play，解决单进程受 GIL 限制无法吃满多核
的问题。架构：

  - 主进程持有训练模型（GPU），每个 cycle 结束后把 CPU state_dict 广播到
    版本化共享字典；
  - 每个 worker 进程持有自己的 CPU 推理模型，开新局前若发现版本更新则
    load_state_dict 同步权重；
  - worker 把整局样本 push 进无界共享队列，主进程按需收集。

> 注意：本模块的进程池必须在 `model.to(cuda)` 之前创建——fork 发生在 CUDA
  初始化前，可避免子进程继承 CUDA 上下文导致的死锁/崩溃（见 train.py 中
  的调用顺序）。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import torch

from src.network.config import MCTSConfig, NetworkConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample
from src.network.selfplay import _self_play_one_game


def _worker_loop(
    shared: Any,
    sample_queue: Any,
    stop_event: Any,
    network_config: NetworkConfig,
    mcts_config: MCTSConfig,
) -> None:
    """worker 主循环：等权重 → 同步 → 跑一局 → push 样本，直到 stop_event。"""
    # 多 worker 并行时避免 CPU 线程池过度订阅
    torch.set_num_threads(1)

    model = AmazonsNet(network_config)
    model.eval()
    last_version = -1

    while not stop_event.is_set():
        version = shared["version"]
        if version == 0:
            # 尚未收到初始权重（主进程还未广播），短暂等待
            time.sleep(0.05)
            continue
        if version != last_version:
            model.load_state_dict(shared["weights"])
            last_version = version

        samples = _self_play_one_game(model, mcts_config)
        sample_queue.put(samples)


class ParallelSelfPlayPool:
    """多进程 self-play 工作池（广播权重 + 收集整局样本）。"""

    def __init__(
        self,
        num_workers: int,
        network_config: NetworkConfig,
        mcts_config: MCTSConfig,
    ) -> None:
        """创建并启动 num_workers 个 self-play worker 进程。

        Args:
            num_workers: worker 进程数（须 > 0）
            network_config: 网络架构（worker 据此构建各自的 CPU 模型）
            mcts_config: MCTS 搜索超参数
        """
        if num_workers <= 0:
            raise ValueError("num_workers 必须 > 0")

        self._manager = mp.Manager()
        self._shared = self._manager.dict()
        self._shared["version"] = 0
        self._shared["weights"] = None
        self._sample_queue: Any = mp.Queue()
        self._stop_event: Any = mp.Event()

        self._workers = [
            mp.Process(
                target=_worker_loop,
                args=(
                    self._shared,
                    self._sample_queue,
                    self._stop_event,
                    network_config,
                    mcts_config,
                ),
                daemon=True,
            )
            for _ in range(num_workers)
        ]
        for worker in self._workers:
            worker.start()

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        """广播最新权重（CPU state_dict）。版本号自增，worker 据此同步。"""
        self._shared["weights"] = state_dict
        self._shared["version"] = self._shared["version"] + 1

    def collect_games(self, num_games: int) -> list[list[Sample]]:
        """阻塞收集 num_games 局整局样本。"""
        games: list[list[Sample]] = []
        while len(games) < num_games:
            games.append(self._sample_queue.get())
        return games

    def shutdown(self) -> None:
        """停止所有 worker 并回收资源。"""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        self._manager.shutdown()
