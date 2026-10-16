"""GPU 批量自我对弈（批量 MCTS，AlphaZero 风格）。

把多条并行的 MCTS 搜索以「锁步」方式推进：每一轮让每条搜索各选出一个待评估
叶子，把所有叶子局面合并成一个 batch 一次送 GPU 前向，再分别展开 + 回传。
GPU 的并行算力因此在 batch 维度（= 并发局数）上得到发挥，而非单局面串行前向。

与 selfplay_parallel.py（多进程 CPU 池）互为替代，由 config 的
`training.self_play_backend`（"gpu" / "cpu"）切换。
"""

from __future__ import annotations

import numpy as np

from src.ai.mcts import (
    MCTSNode,
    _build_root,
    _build_visit_distribution,
    _compute_priors,
    _inject_dirichlet_noise,
    _select_to_leaf,
    decode_move_key,
)
from src.game.board import Board
from src.game.constants import WHITE
from src.game.move import Move
from src.game.rules import Rules
from src.network.config import MCTSConfig
from src.network.model import AmazonsNet
from src.network.replay_buffer import Sample
from src.network.selfplay import DEFAULT_EVAL_OPENING_PLIES


def _batched_search(
    boards: list[Board],
    model: AmazonsNet,
    mcts_config: MCTSConfig,
    temperatures: list[float],
    add_noise_flags: list[bool],
) -> list[tuple[dict[int, float], Move]]:
    """对一批局面各做一次 MCTS 搜索，叶子评估在 GPU 上批量进行。

    Args:
        boards: 并发搜索的局面（各自视角独立，均为非终局）
        model: 目标设备（通常 GPU）上的网络
        mcts_config: MCTS 超参数
        temperatures: 与 boards 对齐的采样温度
        add_noise_flags: 与 boards 对齐的是否注入 Dirichlet 噪声

    Returns:
        与 boards 对齐的 (π, 采样走法) 列表
    """
    roots = [_build_root(b) for b in boards]

    # 1) 批量评估所有根节点，设置先验（根价值被忽略，与串行版一致）
    logits_list, _ = model.predict_batch(
        [root.board for root in roots],
        [root.untried_moves for root in roots],
    )
    for root, logits, add_noise in zip(roots, logits_list, add_noise_flags, strict=True):
        root.priors = _compute_priors(root.untried_moves, logits)
        if add_noise:
            _inject_dirichlet_noise(
                root, mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon
            )

    # 2) 逐轮模拟：每轮每条搜索选一个叶子，批量评估后展开 + 回传
    for _ in range(mcts_config.num_simulations):
        pending: list[list[MCTSNode]] = []
        for root in roots:
            path = _select_to_leaf(root, mcts_config.c_puct)
            leaf = path[-1]
            if leaf.is_terminal:
                value = leaf.terminal_value
                for node in reversed(path):
                    node.n_visits += 1
                    node.total_value += value
                    value = -value
            else:
                pending.append(path)

        if pending:
            leaf_boards = [p[-1].board for p in pending]
            leaf_moves = [p[-1].untried_moves for p in pending]
            logits_list, values = model.predict_batch(leaf_boards, leaf_moves)
            for path, logits, value in zip(pending, logits_list, values, strict=True):
                leaf = path[-1]
                leaf.priors = _compute_priors(leaf.untried_moves, logits)
                v = float(value)
                for node in reversed(path):
                    node.n_visits += 1
                    node.total_value += v
                    v = -v

    # 3) 汇总访问分布并采样
    return [
        _build_visit_distribution(root, temperature)
        for root, temperature in zip(roots, temperatures, strict=True)
    ]


class _Simulator:
    """一条自我对弈局面的推进状态（供批量自对弈使用）。"""

    def __init__(self, mcts_config: MCTSConfig) -> None:
        self._mcts_config = mcts_config
        self.board = Board()
        self.move_index = 0
        self.records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self.finished = False
        self.samples: list[Sample] = []

    def temperature(self) -> float:
        """前 30 步用 self_play_temperature，之后贪心。"""
        return (
            self._mcts_config.self_play_temperature if self.move_index < 30 else 0.0
        )

    def add_noise(self) -> bool:
        """前 30 步注入 Dirichlet 噪声。"""
        return self.move_index < 30 and self._mcts_config.dirichlet_epsilon > 0.0

    def step(self, pi: dict[int, float], move: Move) -> None:
        """记录本步样本并推进棋盘；若终局则结算整局样本。"""
        keys = list(pi.keys())
        moves = np.array([decode_move_key(k) for k in keys], dtype=np.int16)
        probs = np.array([pi[k] for k in keys], dtype=np.float32)
        state = AmazonsNet.board_to_tensor(self.board).astype(np.int8)
        self.records.append((state, moves, probs, self.board.get_current_player()))
        self.board.make_move(move)
        self.move_index += 1

        if Rules.is_game_over(self.board):
            self.finished = True
            winner = Rules.get_winner(self.board)
            self.samples = [
                (s, m, p, 1.0 if player == winner else -1.0)
                for (s, m, p, player) in self.records
            ]


def generate_self_play_data_gpu(
    model: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
    batch_size: int = 64,
) -> list[list[Sample]]:
    """GPU 批量自我对弈，生成训练数据。

    每批 `batch_size` 局并行推进（锁步），批内全部结束后进入下一批；批内
    叶子评估在 GPU 上合并成一个 batch 前向。

    Args:
        model: 目标设备（通常 GPU）上的网络
        num_games: 对弈局数
        mcts_config: MCTS 搜索超参数
        batch_size: 并发局数（= 每次 GPU 前向的 board 数）

    Returns:
        每局一个样本列表
    """
    samples_all: list[list[Sample]] = []
    games_made = 0

    while games_made < num_games:
        chunk_size = min(batch_size, num_games - games_made)
        sims = [_Simulator(mcts_config) for _ in range(chunk_size)]
        games_made += chunk_size

        active = [s for s in sims if not s.finished]
        while active:
            results = _batched_search(
                [s.board for s in active],
                model,
                mcts_config,
                [s.temperature() for s in active],
                [s.add_noise() for s in active],
            )
            for sim, (pi, move) in zip(active, results, strict=True):
                sim.step(pi, move)
            active = [s for s in sims if not s.finished]

        for sim in sims:
            samples_all.append(sim.samples)

    return samples_all


def play_eval_games_gpu(
    model_a: AmazonsNet,
    model_b: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
    batch_size: int = 16,
    opening_plies: int = DEFAULT_EVAL_OPENING_PLIES,
    first_game_a_is_white: bool = True,
) -> list[tuple[int, bool]]:
    """A/B 对战评估（批量、双模型），返回每局的 (胜方, A是否执白)。

    口径与 selfplay.pit_evaluate 一致：各执一半白/黑（抵消先手优势），
    前 opening_plies 步按访问分布采样（temperature=1）以随机化开局，之后贪心。
    随机化是必须的——全程贪心时同一对模型每局走法完全相同，N 局的信息量会
    退化成 1 局。

    与串行评估的区别：这里把一批对局以锁步方式并行推进，每步按"该谁走"把
    局面分成两组、各做一次批量 MCTS。因此评估速度提升约等于并发度，让评估
    从整个训练闭环里最贵的一环（串行 40 局约 40 分钟）变成可忽略的开销。

    Args:
        model_a: A 方网络（目标设备上）
        model_b: B 方网络（目标设备上，可与 A 同卡）
        num_games: 对局数
        mcts_config: MCTS 超参数
        batch_size: 并发推进的局数
        opening_plies: 开局随机化步数
        first_game_a_is_white: 第一局 A 是否执白；同批内交替先后手。
            多 worker 时分奇偶传入，可保证全局黑白局数平衡。

    Returns:
        [(winner, a_is_white), ...]，winner 为 WHITE/BLACK
    """

    class _EvalGame:
        """一条评估对局的推进状态。"""

        def __init__(self, a_is_white: bool) -> None:
            self.a_is_white = a_is_white
            self.board: Board = Board()
            self.ply = 0
            self.finished = False
            self.winner = 0

        def actor_is_a(self) -> bool:
            """当前该走的是否 A 方。"""
            return (self.board.get_current_player() == WHITE) == self.a_is_white

        def temperature(self) -> float:
            """开局若干步采样，之后贪心。"""
            return 1.0 if self.ply < opening_plies else 0.0

        def step(self, move: Move) -> None:
            self.board.make_move(move)
            self.ply += 1
            if Rules.is_game_over(self.board):
                self.finished = True
                self.winner = Rules.get_winner(self.board)

    results: list[tuple[int, bool]] = []
    made = 0
    while made < num_games:
        chunk = min(batch_size, num_games - made)
        # 交替先后手：A 执白一局、执黑一局
        batch: list[_EvalGame] = []
        for i in range(chunk):
            a_is_white = (i % 2 == 0) if first_game_a_is_white else (i % 2 == 1)
            batch.append(_EvalGame(a_is_white=a_is_white))
        made += chunk

        active = [g for g in batch if not g.finished]
        while active:
            a_games = [g for g in active if g.actor_is_a()]
            b_games = [g for g in active if not g.actor_is_a()]
            if a_games:
                moves_a = _batched_search(
                    [g.board for g in a_games],
                    model_a,
                    mcts_config,
                    [g.temperature() for g in a_games],
                    [False] * len(a_games),
                )
                for game, (_, move) in zip(a_games, moves_a, strict=True):
                    game.step(move)
            if b_games:
                moves_b = _batched_search(
                    [g.board for g in b_games],
                    model_b,
                    mcts_config,
                    [g.temperature() for g in b_games],
                    [False] * len(b_games),
                )
                for game, (_, move) in zip(b_games, moves_b, strict=True):
                    game.step(move)
            active = [g for g in batch if not g.finished]

        results.extend((g.winner, g.a_is_white) for g in batch)

    return results
