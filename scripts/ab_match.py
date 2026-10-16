#!/usr/bin/env python3
"""A/B 对战脚本——把两个引擎放在同一批开局下配对对战，输出胜率与置信区间。

为什么这样设计：

1. **配对开局（paired openings）**：先用固定随机种子生成一批开局走法序列，
   两个引擎从**完全相同的局面**开始对弈。开局运气因此被消除。
2. **交换先后手**：每个开局下下两局，A 执白一局、执黑一局。亚马逊棋白方有
   先手优势，交换后该优势抵消。两局合起来算一个"配对"。
3. **置信区间**：20 局无法区分 55% 和 50%（信噪比约 1:1）。本脚本用 Wilson
   区间给出 95% 置信区间，并明确告诉你差异是否显著——避免拿噪声当结论。

用法示例::

    # 修好的 alpha-beta（depth 3）对 MCTS（现有 best_model，150 模拟）
    python scripts/ab_match.py --a alphabeta:3 --b mcts:models/best_model.pt:150 \\
        --openings 50 --time-budget-ms 2000

    # 两个 checkpoint 互相对战
    python scripts/ab_match.py --a mcts:models/checkpoint_cycle10.pt:150 \\
        --b mcts:models/best_model.pt:150 --openings 50

    # 冒烟测试（很快，只验证脚本能跑通）
    python scripts/ab_match.py --a random --b random --openings 3

引擎写法（--a / --b）::

    random                          随机走法（基线）
    alphabeta:DEPTH                 alpha-beta 搜索，如 alphabeta:3
    mcts:MODEL_PATH:SIMS            网络引导 MCTS，如 mcts:models/best_model.pt:150
                                    （默认按 4块/128 加载）
    mcts:MODEL_PATH:SIMS:BLOCKSxFILTERS
                                   指定模型结构，如 ...:400:13x256——
                                   训练出的正式模型必须这么写，否则 size mismatch

运行前请清掉 ROS 的 PYTHONPATH（否则 pytest/torch 环境会互相污染）::

    PYTHONPATH= python scripts/ab_match.py ...
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# 允许从任意目录运行（scripts/ 的上一级才是项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.board import Board  # noqa: E402
from src.game.constants import WHITE  # noqa: E402
from src.game.move import Move  # noqa: E402
from src.game.rules import Rules  # noqa: E402

# 一个"搜索引擎函数"：给棋盘和时间预算，返回走法
SearchFn = Callable[[Board, float], Move]


@dataclass(frozen=True)
class EngineSpec:
    """解析后的引擎描述。"""

    raw: str
    kind: str  # "random" | "alphabeta" | "mcts"
    depth: int = 0
    model_path: str = ""
    sims: int = 0
    blocks: int = 4          # 模型残差块数（现有旧权重是 4）
    filters: int = 128       # 模型滤波器数（现有旧权重是 128）

    def label(self) -> str:
        """用于输出的简短标签。"""
        if self.kind == "alphabeta":
            return f"alphabeta(depth={self.depth})"
        if self.kind == "mcts":
            return (
                f"mcts({Path(self.model_path).name}, {self.sims} sims, "
                f"{self.blocks}块/{self.filters}滤波)"
            )
        return "random"


@dataclass
class GameOutcome:
    """一局的结果。"""

    winner: int | None  # WHITE / BLACK；None 表示未终局
    plies: int
    elapsed_s: float
    error: str = ""


def parse_engine_spec(spec: str) -> EngineSpec:
    """解析引擎描述字符串。

    Args:
        spec: 形如 ``random`` / ``alphabeta:3`` / ``mcts:path/to/model.pt:150``

    Returns:
        EngineSpec

    Raises:
        ValueError: 描述格式无法识别
    """
    parts = spec.split(":")
    kind = parts[0].strip().lower()

    if kind == "random":
        return EngineSpec(raw=spec, kind="random")

    if kind == "alphabeta":
        if len(parts) != 2:
            raise ValueError(f"alphabeta 需要写成 alphabeta:DEPTH，收到 '{spec}'")
        return EngineSpec(raw=spec, kind="alphabeta", depth=int(parts[1]))

    if kind == "mcts":
        if len(parts) not in (3, 4):
            raise ValueError(
                f"mcts 需写成 mcts:MODEL_PATH:SIMS 或 "
                f"mcts:MODEL_PATH:SIMS:BLOCKSxFILTERS，收到 '{spec}'"
            )
        blocks, filters = 4, 128
        if len(parts) == 4:
            # 例如 13x256 —— 训练出的正式模型是 13 块 / 256 滤波
            try:
                b_str, f_str = parts[3].lower().split("x")
                blocks, filters = int(b_str), int(f_str)
            except ValueError as e:
                raise ValueError(
                    f"结构需写成 BLOCKSxFILTERS（如 13x256），收到 '{parts[3]}'"
                ) from e
        return EngineSpec(
            raw=spec, kind="mcts", model_path=parts[1], sims=int(parts[2]),
            blocks=blocks, filters=filters,
        )

    raise ValueError(f"无法识别的引擎类型 '{kind}'（可用: random / alphabeta / mcts）")


def build_search_fn(spec: EngineSpec) -> SearchFn:
    """把引擎描述实例化为一个 search(board, budget_ms) -> Move 函数。

    torch 相关模块延迟导入，只有真正用到 MCTS 时才需要 torch。

    Args:
        spec: 引擎描述

    Returns:
        搜索函数

    Raises:
        RuntimeError: 模型文件缺失或依赖不可用
    """
    if spec.kind == "random":

        def random_search(board: Board, _budget_ms: float) -> Move:
            return random.choice(Rules.get_legal_moves(board))

        return random_search

    if spec.kind == "alphabeta":
        from src.ai.alphabeta import AlphaBetaEngine
        from src.ai.evaluator import HeuristicEvaluator

        engine = AlphaBetaEngine(
            evaluator=HeuristicEvaluator(),
            max_depth=spec.depth,
            time_budget_ms=float("inf"),
            use_transposition_table=True,
            use_iterative_deepening=True,
        )

        def alphabeta_search(board: Board, budget_ms: float) -> Move:
            return engine.search(board, budget_ms)

        return alphabeta_search

    # MCTS
    if not Path(spec.model_path).is_file():
        raise RuntimeError(f"模型文件不存在: {spec.model_path}")

    from src.ai.mcts import MCTSEngine
    from src.network.config import NetworkConfig
    from src.network.train import load_checkpoint

    # 结构由引擎描述里的第 4 段给出（如 mcts:path:400:13x256）；
    # 不写则按 4块/128（现有旧权重的规模）。写错会 size mismatch。
    net_config = NetworkConfig(
        num_residual_blocks=spec.blocks, num_filters=spec.filters
    )
    model = load_checkpoint(spec.model_path, net_config)

    # 有 CUDA 就上显卡，没有就退回 CPU
    try:
        import torch

        if torch.cuda.is_available():
            model = model.to("cuda")
    except ImportError:
        pass

    mcts_engine = MCTSEngine(
        model, num_simulations=spec.sims, c_puct=1.0, temperature=0.0
    )

    def mcts_search(board: Board, budget_ms: float) -> Move:
        return mcts_engine.search(board, budget_ms)

    return mcts_search

def generate_openings(
    num_openings: int, plies: int, seed: int
) -> list[list[Move]]:
    """生成一批随机开局走法序列。

    同一批开局会用于 A/B 双方，且每个开局下交换先后手各下一局，因此开局的
    运气成分被抵消。

    Args:
        num_openings: 开局数量（= 配对数）
        plies: 每个开局的步数
        seed: 随机种子（固定后结果可复现）

    Returns:
        开局序列列表
    """
    rng = random.Random(seed)
    openings: list[list[Move]] = []
    for _ in range(num_openings):
        board = Board()
        seq: list[Move] = []
        for _ in range(plies):
            moves = Rules.get_legal_moves(board)
            if not moves:
                break
            move = rng.choice(moves)
            seq.append(move)
            board.make_move(move)
        openings.append(seq)
    return openings


def play_game(
    white_fn: SearchFn,
    black_fn: SearchFn,
    opening: list[Move],
    time_budget_ms: float,
    max_plies: int,
) -> GameOutcome:
    """按给定开局下一局，返回结果。

    引擎抛异常或返回非法走法时，该引擎判负（并记录原因），而不是让整个
    对比跑挂掉。

    Args:
        white_fn: 执白引擎的搜索函数
        black_fn: 执黑引擎的搜索函数
        opening: 开局走法序列
        time_budget_ms: 每步时间预算
        max_plies: 步数上限（超出记为未终局，不计入胜负）

    Returns:
        GameOutcome
    """
    board = Board()
    for move in opening:
        if move not in set(Rules.get_legal_moves(board)):
            # 生成器保证合法；这里防御性检查，避免污染棋盘状态
            return GameOutcome(winner=None, plies=0, elapsed_s=0.0, error="开局走法非法")
        board.make_move(move)

    plies = len(opening)
    start = time.time()

    while not Rules.is_game_over(board):
        if plies >= max_plies:
            return GameOutcome(
                winner=None,
                plies=plies,
                elapsed_s=time.time() - start,
                error="达到步数上限",
            )

        current = board.get_current_player()
        search_fn = white_fn if current == WHITE else black_fn
        try:
            move = search_fn(board.clone(), time_budget_ms)
        except Exception as e:  # noqa: BLE001 - 任何异常都判该引擎负，避免中断整轮
            return GameOutcome(
                winner=-current,
                plies=plies,
                elapsed_s=time.time() - start,
                error=f"引擎异常: {type(e).__name__}: {e}",
            )

        if move not in set(Rules.get_legal_moves(board)):
            return GameOutcome(
                winner=-current,
                plies=plies,
                elapsed_s=time.time() - start,
                error=f"引擎返回非法走法: {move.to_string()}",
            )

        board.make_move(move)
        plies += 1

    return GameOutcome(
        winner=Rules.get_winner(board), plies=plies, elapsed_s=time.time() - start
    )


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间（比正态近似更适合接近 0/1 的比例）。

    Args:
        wins: 获胜次数
        n: 总场次
        z: 置信水平系数，1.96 ≈ 95%

    Returns:
        (下界, 上界)
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="A/B 对战：两个引擎在相同开局下配对对战，输出胜率与置信区间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--a", required=True, help="引擎 A 描述（见文件头说明）")
    parser.add_argument("--b", required=True, help="引擎 B 描述（见文件头说明）")
    parser.add_argument(
        "--openings", type=int, default=50, help="开局数（每局开下 2 盘，默认 50）"
    )
    parser.add_argument(
        "--opening-plies", type=int, default=6, help="每个开局的随机步数（默认 6）"
    )
    parser.add_argument(
        "--time-budget-ms",
        type=float,
        default=2000.0,
        help="每步时间预算（毫秒，默认 2000）",
    )
    parser.add_argument(
        "--max-plies", type=int, default=400, help="单局步数上限（默认 400）"
    )
    parser.add_argument("--seed", type=int, default=20261015, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """入口。"""
    args = parse_args()

    spec_a = parse_engine_spec(args.a)
    spec_b = parse_engine_spec(args.b)
    label_a, label_b = spec_a.label(), spec_b.label()

    print("=" * 72)
    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print(
        f"开局数 {args.openings} × 2 盘（交换先后手）= {args.openings * 2} 局 | "
        f"每步 {args.time_budget_ms:.0f}ms | 开局 {args.opening_plies} 步 | "
        f"seed {args.seed}"
    )
    print("=" * 72, flush=True)

    fn_a = build_search_fn(spec_a)
    fn_b = build_search_fn(spec_b)

    openings = generate_openings(args.openings, args.opening_plies, args.seed)

    a_wins = 0
    b_wins = 0
    unfinished = 0
    errors: list[str] = []
    total_plies = 0
    decided = 0
    # 分先后手统计，用于检查是否存在颜色偏差
    a_wins_as_white = a_wins_as_black = 0
    games_as_white = games_as_black = 0
    start_all = time.time()

    for idx, opening in enumerate(openings, start=1):
        # 一个配对：A 执白一局、A 执黑一局
        for a_is_white in (True, False):
            outcome = play_game(
                fn_a if a_is_white else fn_b,
                fn_b if a_is_white else fn_a,
                opening,
                args.time_budget_ms,
                args.max_plies,
            )
            total_plies += outcome.plies
            if outcome.error:
                errors.append(outcome.error)
            if a_is_white:
                games_as_white += 1
            else:
                games_as_black += 1

            if outcome.winner is None:
                unfinished += 1
                verdict = "未终局"
            else:
                decided += 1
                a_won = (outcome.winner == WHITE) == a_is_white
                if a_won:
                    a_wins += 1
                    if a_is_white:
                        a_wins_as_white += 1
                    else:
                        a_wins_as_black += 1
                else:
                    b_wins += 1
                verdict = f"A{'胜' if a_won else '负'}"

            print(
                f"[{idx:3d}/{args.openings}] A执{'白' if a_is_white else '黑'} "
                f"{verdict}  {outcome.plies:3d}步 {outcome.elapsed_s:5.1f}s"
                + (f"  ⚠ {outcome.error}" if outcome.error else ""),
                flush=True,
            )

    elapsed = time.time() - start_all

    print()
    print("=" * 72)
    print(f"A({label_a})  {a_wins} : {b_wins}  B({label_b})")
    if unfinished:
        print(f"未计入胜负（未终局）: {unfinished} 局")
    if decided:
        rate = a_wins / decided
        lo, hi = wilson_interval(a_wins, decided)
        print(f"A 胜率 = {rate:.3f}   95% 置信区间 [{lo:.3f}, {hi:.3f}]")
        print(
            f"分先后手: A执白 {a_wins_as_white}/{games_as_white} 胜, "
            f"A执黑 {a_wins_as_black}/{games_as_black} 胜"
        )
        print()
        if lo > 0.5:
            print(f"结论: A 显著强于 B（置信区间下界 {lo:.3f} > 0.5）")
        elif hi < 0.5:
            print(f"结论: B 显著强于 A（置信区间上界 {hi:.3f} < 0.5）")
        else:
            need = math.ceil(1.96**2 * 0.25 / (0.05**2)) if rate == 0.5 else 0
            print(
                f"结论: 无显著差异（0.5 落在 [{lo:.3f}, {hi:.3f}] 内）"
                f" —— 当前 {decided} 局的样本量不足以区分两者。"
            )
            if need:
                print(f"      若真实胜率就是 0.5，检出 5% 差异约需 {need} 局以上。")
    total_games = max(args.openings * 2, 1)
    print(
        f"平均步数 {total_plies / total_games:.1f} | 总耗时 {elapsed / 60:.1f} 分钟"
    )
    if errors:
        kinds: dict[str, int] = {}
        for err in errors:
            kinds[err] = kinds.get(err, 0) + 1
        print("异常统计:")
        for err, cnt in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt:4d} × {err}")
    print("=" * 72)


if __name__ == "__main__":
    main()
