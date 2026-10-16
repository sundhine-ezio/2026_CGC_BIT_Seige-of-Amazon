#!/usr/bin/env python3
"""自对弈吞吐基准——用于在目标机器上实测并据此决定并行 worker 数。

为什么需要它：自对弈的瓶颈是 **CPU 单核的走法生成**，不是显卡。实测（RTX
4060 Laptop，4块/128 小模型，batch 4）GPU 利用率平均仅 6.4%、显存 3%。所以
"换更强的显卡"本身不会加速；加速来自**并行跑 N 个 self-play 进程**，而 N
取多少取决于显卡在多少个进程下被喂饱——**这个只能实测**。

三种模式：

1. ``--backend gpu``    单进程批量 MCTS 的吞吐与 GPU 占用（基线）
2. ``--backend pool``   多进程 GPU 池，指定 ``--workers`` 的聚合吞吐
3. ``--sweep 8,16,32``  对多个 worker 数依次测量，直接输出"该用几个"的结论

模型结构用 ``--blocks/--filters`` 指定；若 checkpoint 与结构不匹配（例如拿
4块/128 的权重去测正式训练的 13块/256），会自动退化为**随机初始化**——吞吐
只取决于计算量，与权重数值无关，所以测出来的数字依然有效。

用法::

    # 基线：单进程，正式网络结构（随机初始化）
    PYTHONPATH= python scripts/bench_selfplay.py --blocks 13 --filters 256 \\
        --games 8 --sims 200 --batch 8 --device cuda:0

    # 直接给出 worker 数结论（推荐；双卡）
    PYTHONPATH= python scripts/bench_selfplay.py --sweep 8,16,24,32,48 \\
        --blocks 13 --filters 256 --devices 0,1 \\
        --sims 200 --batch 8 --games-per-worker 2

    # 对比 CPU 多进程后端
    PYTHONPATH= python scripts/bench_selfplay.py --backend cpu --workers 32 \\
        --blocks 13 --filters 256 --sims 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network.config import MCTSConfig, NetworkConfig  # noqa: E402


def build_net_config(args: argparse.Namespace) -> NetworkConfig:
    """按命令行参数构建网络结构配置。"""
    return NetworkConfig(
        num_residual_blocks=args.blocks,
        num_filters=args.filters,
    )


def build_mcts_config(args: argparse.Namespace) -> MCTSConfig:
    """按命令行参数构建 MCTS 配置。"""
    return MCTSConfig(
        num_simulations=args.sims,
        c_puct=1.0,
        temperature=0.0,
        self_play_temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )


def make_weights(net_config: NetworkConfig, args: argparse.Namespace) -> dict:
    """准备一份 CPU state_dict 供广播。

    优先加载 checkpoint；若文件不存在或结构与 ``--blocks/--filters`` 不匹配
    （例如拿 4块/128 的权重测 13块/256），退化为随机初始化。

    Args:
        net_config: 目标网络结构
        args: 命令行参数

    Returns:
        CPU state_dict
    """
    from src.network.model import AmazonsNet

    if args.model and Path(args.model).is_file():
        try:
            from src.network.train import load_checkpoint

            model = load_checkpoint(args.model, net_config)
            print(f"已加载权重 {args.model}")
            return {k: v.detach().clone() for k, v in model.state_dict().items()}
        except Exception as e:  # noqa: BLE001 - 结构不匹配就退化为随机初始化
            print(
                f"⚠ 加载 {args.model} 失败（{type(e).__name__}），"
                f"退化为随机初始化——吞吐不受影响",
                flush=True,
            )
    else:
        print("未指定/未找到 checkpoint，使用随机初始化权重")

    model = AmazonsNet(net_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M")
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class GpuSampler:
    """后台采样所有 GPU 的利用率与显存。"""

    def __init__(self) -> None:
        self.samples: list[tuple[int, float, float]] = []
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._done.is_set():
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    for line in r.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 3:
                            self.samples.append(
                                (int(parts[0]), float(parts[1]), float(parts[2]))
                            )
            except Exception:
                pass
            time.sleep(0.3)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=2)

    def report(self) -> dict[int, float]:
        """打印每张卡的统计，返回 {卡号: 平均利用率}。"""
        if not self.samples:
            print("（没采到 GPU 数据）")
            return {}
        by_gpu: dict[int, list[tuple[float, float]]] = {}
        for idx, util, mem in self.samples:
            by_gpu.setdefault(idx, []).append((util, mem))
        avg_util: dict[int, float] = {}
        for idx in sorted(by_gpu):
            utils = [u for u, _ in by_gpu[idx]]
            mems = [m for _, m in by_gpu[idx]]
            avg = sum(utils) / len(utils)
            avg_util[idx] = avg
            print(
                f"  GPU{idx}: 利用率 平均 {avg:5.1f}% 最大 {max(utils):3.0f}% | "
                f"显存峰值 {max(mems):6.0f} MiB"
            )
        return avg_util


def run_gpu_single(args: argparse.Namespace, net_config: NetworkConfig, weights: dict) -> None:
    """模式 1：单进程批量 MCTS。"""
    import torch

    from src.network.model import AmazonsNet
    from src.network.selfplay_gpu import generate_self_play_data_gpu

    if not torch.cuda.is_available():
        print("CUDA 不可用", file=sys.stderr)
        sys.exit(1)
    device_index = int(args.device.split(":")[1]) if ":" in args.device else 0
    props = torch.cuda.get_device_properties(device_index)
    print(f"显卡 [{device_index}] {props.name}  {props.total_memory / 2**20:.0f} MiB")

    model = AmazonsNet(net_config).to(args.device)
    model.load_state_dict(weights)
    model.eval()

    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    games = generate_self_play_data_gpu(
        model, args.games, build_mcts_config(args), args.batch
    )
    elapsed = time.time() - t0
    sampler.stop()

    samples = sum(len(g) for g in games)
    sims = samples * args.sims
    print()
    print(f"完成 {len(games)} 局，样本 {samples}，耗时 {elapsed:.1f}s")
    print(f"**吞吐 {sims / elapsed:.0f} 次模拟/秒**（单进程）")
    avg = sampler.report()
    if avg:
        a = list(avg.values())[0]
        print(f"  → 按平均利用率 {a:.1f}% 估算，约需 {max(1, round(100 / a))} "
              f"个进程才能喂饱这张卡")


def run_pool(
    args: argparse.Namespace,
    net_config: NetworkConfig,
    weights: dict,
    workers: int,
    devices: list[int],
) -> float:
    """模式 2/3：多进程 GPU 池，返回聚合吞吐（次模拟/秒）。"""
    from src.network.selfplay_gpu_pool import GpuSelfPlayPool

    t_create = time.time()
    pool = GpuSelfPlayPool(
        workers,
        net_config,
        build_mcts_config(args),
        devices,
        batch_size=args.batch,
        games_per_round=args.games_per_worker,
    )
    create_s = time.time() - t_create

    sampler = GpuSampler()
    try:
        sampler.start()
        # 先广播（worker 才会开工），再计时——排除建池与 CUDA 初始化的开销
        t0 = time.time()
        pool.broadcast_weights(weights)
        games = pool.collect_games(0)
        elapsed = time.time() - t0
        sampler.stop()
    finally:
        pool.shutdown()

    samples = sum(len(g) for g in games)
    sims = samples * args.sims
    throughput = sims / elapsed
    print(
        f"  {workers:3d} workers | {len(games):4d} 局 | {samples:6d} 样本 | "
        f"{elapsed:7.1f}s | **{throughput:7.0f} 次模拟/秒** | "
        f"建池 {create_s:.1f}s"
    )
    sampler.report()
    return throughput


def run_cpu_pool(
    args: argparse.Namespace, net_config: NetworkConfig, workers: int
) -> float:
    """模式：多进程 CPU 池（网络在 CPU 上推理）。"""
    from src.network.selfplay_parallel import ParallelSelfPlayPool

    pool = ParallelSelfPlayPool(workers, net_config, build_mcts_config(args))
    try:
        from src.network.model import AmazonsNet

        model = AmazonsNet(net_config)
        pool.broadcast_weights(
            {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        t0 = time.time()
        games = pool.collect_games(args.games)
        elapsed = time.time() - t0
    finally:
        pool.shutdown()

    samples = sum(len(g) for g in games)
    sims = samples * args.sims
    throughput = sims / elapsed
    print(
        f"  CPU {workers:3d} workers | {len(games):4d} 局 | {elapsed:7.1f}s | "
        f"**{throughput:7.0f} 次模拟/秒**"
    )
    return throughput


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="自对弈吞吐基准")
    p.add_argument(
        "--backend",
        choices=["gpu", "pool", "cpu"],
        default="gpu",
        help="gpu=单进程批量 MCTS；pool=多进程 GPU 池；cpu=多进程 CPU 池",
    )
    p.add_argument("--model", default="", help="checkpoint 路径（空=随机初始化）")
    p.add_argument("--games", type=int, default=8, help="单进程模式的局数")
    p.add_argument("--sims", type=int, default=200, help="每步 MCTS 模拟次数")
    p.add_argument("--batch", type=int, default=8, help="每 worker 并发推进的局数")
    p.add_argument("--workers", type=int, default=16, help="pool/cpu 模式的 worker 数")
    p.add_argument(
        "--sweep",
        default="",
        help='逗号分隔的 worker 数列表，如 "8,16,32"——依次测量并给出结论',
    )
    p.add_argument(
        "--games-per-worker",
        type=int,
        default=2,
        help="pool 模式每个 worker 每轮跑几局（一轮总产量 = workers × 该值）",
    )
    p.add_argument("--devices", default="0,1", help='pool 模式用的 GPU，如 "0,1"')
    p.add_argument("--device", default="cuda:0", help="单进程模式用的设备")
    p.add_argument("--blocks", type=int, default=4, help="残差块数")
    p.add_argument("--filters", type=int, default=128, help="滤波器数")
    return p.parse_args()


def main() -> None:
    """入口。"""
    args = parse_args()
    net_config = build_net_config(args)

    print("=" * 78)
    print(
        f"网络 {args.blocks}块/{args.filters}滤波 | 每步 {args.sims} 模拟 | "
        f"batch {args.batch} | 后端 {args.backend}"
    )
    print("=" * 78, flush=True)

    weights = make_weights(net_config, args)
    print()

    if args.sweep:
        from src.network.selfplay_gpu_pool import parse_devices

        devices = parse_devices(args.devices)
        worker_counts = [int(x) for x in args.sweep.replace(" ", "").split(",") if x]
        print(f"扫描 worker 数 {worker_counts}，GPU {devices}，"
              f"每 worker {args.games_per_worker} 局/轮")
        print()
        results: list[tuple[int, float]] = []
        for n in worker_counts:
            results.append((n, run_pool(args, net_config, weights, n, devices)))
        print()
        print("=" * 78)
        base = results[0][1]
        print(f"{'workers':>8} {'吞吐(模拟/秒)':>14} {'相对首个':>9}")
        best = max(results, key=lambda kv: kv[1])
        for n, t in results:
            mark = "  ← 最佳" if (n, t) == best else ""
            print(f"{n:>8} {t:>14.0f} {t / base:>8.2f}×{mark}")
        print()
        print(f"建议起始 worker 数: **{best[0]}**（吞吐 {best[1]:.0f} 次模拟/秒）")
        print("说明：再加 worker 若吞吐不再上升，说明显卡已饱和；")
        print("      若吞吐随 worker 数下降，说明 CPU 或显存出现争抢。")
        return

    if args.backend == "gpu":
        run_gpu_single(args, net_config, weights)
    elif args.backend == "pool":
        from src.network.selfplay_gpu_pool import parse_devices

        run_pool(args, net_config, weights, args.workers, parse_devices(args.devices))
    else:
        run_cpu_pool(args, net_config, args.workers)


if __name__ == "__main__":
    main()
