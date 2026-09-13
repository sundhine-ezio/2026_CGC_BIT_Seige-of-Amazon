"""蒙特卡洛树搜索（MCTS）。

实现两种 MCTS 引擎：
  - MCTSEngine:     网络引导的 MCTS（AlphaZero 风格），使用策略-价值双头
                    网络提供先验概率与局面价值评估。
  - PureMCTSEngine: 无网络的 MCTS 基线（UCB1 + 随机模拟），用于棋力对照。

两者均实现 Engine 接口，与 AlphaBetaEngine 可互换。

> 实现采用「懒展开」：节点经网络评估后仅保存先验概率与未物化走法列表，
  子节点在被选中时才 clone 棋盘并生成其合法走法，避免一次展开即克隆全部
  （2000+）子棋盘。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.ai.engine import Engine
from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules


class _PolicyValueNet(Protocol):
    """网络模型协议：MCTS 只需 predict 接口，解耦对 PyTorch 的硬依赖。"""

    def predict(self, board: Board, moves: list[Move]) -> tuple[np.ndarray, float]:
        """返回 (每个合法走法的打分 logits, 价值 v ∈ [-1, 1])（当前行棋方视角）。"""
        ...


# =============================================================================
# 走法键编码
# =============================================================================


def move_key(move: Move) -> int:
    """将走法编码为唯一整数键（from/to/arrow 各 2 位十进制，均 < 100）。"""
    fr, fc = move.from_pos
    tr, tc = move.to_pos
    ar, ac = move.arrow_pos
    return (fr * 10 + fc) * 10000 + (tr * 10 + tc) * 100 + (ar * 10 + ac)


def decode_move_key(key: int) -> tuple[int, int, int]:
    """从整数键解码 (from_idx, to_idx, arrow_idx)，每个 ∈ [0, 100)。"""
    from_idx: int = key // 10000
    to_idx: int = (key // 100) % 100
    arrow_idx: int = key % 100
    return from_idx, to_idx, arrow_idx


# =============================================================================
# 树节点与辅助函数
# =============================================================================


@dataclass
class MCTSNode:
    """MCTS 搜索树节点。

    视角约定：total_value 与 terminal_value 始终以「本节点当前行棋方」视角
    记录；回传时每上溯一层取相反数完成视角翻转。
    """

    board: Board
    move: Move | None
    parent: MCTSNode | None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    untried_moves: list[Move] = field(default_factory=list)
    priors: dict[int, float] = field(default_factory=dict)
    n_visits: int = 0
    total_value: float = 0.0
    is_terminal: bool = False
    terminal_value: float = 0.0

    def q_value(self) -> float:
        """平均价值 Q = W / N；N == 0 时返回 0.0。"""
        if self.n_visits == 0:
            return 0.0
        return self.total_value / self.n_visits


def _build_root(board: Board) -> MCTSNode:
    """构建根节点（clone 棋盘并生成合法走法）。"""
    root = MCTSNode(board=board.clone(), move=None, parent=None)
    root.untried_moves = Rules.get_legal_moves(root.board)
    if not root.untried_moves:
        root.is_terminal = True
        root.terminal_value = -1.0
    return root


def _make_child(parent: MCTSNode, move: Move) -> MCTSNode:
    """物化一个子节点：clone 父棋盘、执行走法、生成子节点的合法走法。"""
    child_board = parent.board.clone()
    child_board.make_move(move)
    child = MCTSNode(board=child_board, move=move, parent=parent)
    child.untried_moves = Rules.get_legal_moves(child_board)
    if not child.untried_moves:
        child.is_terminal = True
        child.terminal_value = -1.0
    return child


def _compute_priors(moves: list[Move], policy_logits: np.ndarray) -> dict[int, float]:
    """由走法打分 logits 计算每个合法走法的先验（softmax）。

    policy_logits 与 moves 一一对应（长度相等）；直接对全量合法走法做
    softmax，无需再逐 (from,to) 求皇后可达域掩码，也无需逐箭 clone 棋盘。
    """
    logits = np.asarray(policy_logits, dtype=np.float64)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return {move_key(move): float(probs[i]) for i, move in enumerate(moves)}


def _select_child_puct(node: MCTSNode, c_puct: float) -> MCTSNode:
    """PUCT 选择：在已物化子节点与未物化走法中选择最优者（并物化）。"""
    sqrt_parent = math.sqrt(max(node.n_visits, 1))
    best_score = -float("inf")
    best_child: MCTSNode | None = None
    best_untried: Move | None = None

    for key, child in node.children.items():
        q_parent = -child.q_value()  # 子节点价值为对方视角，翻转到本节点
        prior = node.priors.get(key, 0.0)
        score = q_parent + c_puct * prior * sqrt_parent / (1.0 + child.n_visits)
        if score > best_score:
            best_score = score
            best_child = child
            best_untried = None

    for move in node.untried_moves:
        key = move_key(move)
        prior = node.priors.get(key, 0.0)
        score = c_puct * prior * sqrt_parent  # Q=0，N=0
        if score > best_score:
            best_score = score
            best_child = None
            best_untried = move

    if best_untried is not None:
        child = _make_child(node, best_untried)
        node.children[move_key(best_untried)] = child
        node.untried_moves.remove(best_untried)
        return child

    if best_child is None:
        raise RuntimeError("select_child() called on a childless node")
    return best_child


def _select_to_leaf(root: MCTSNode, c_puct: float) -> list[MCTSNode]:
    """从根一路 PUCT 选择到「待评估」节点（未展开叶子或终局），返回路径。"""
    node: MCTSNode = root
    path: list[MCTSNode] = [root]
    while not node.is_terminal and node.priors and (node.children or node.untried_moves):
        node = _select_child_puct(node, c_puct)
        path.append(node)
    return path


def _inject_dirichlet_noise(
    root: MCTSNode, dirichlet_alpha: float, dirichlet_epsilon: float
) -> None:
    """在根节点先验上注入 Dirichlet 噪声（self-play 探索）。"""
    keys = list(root.priors.keys())
    if not keys:
        return
    noise = np.random.dirichlet([dirichlet_alpha] * len(keys))
    eps = dirichlet_epsilon
    for i, key in enumerate(keys):
        root.priors[key] = (1.0 - eps) * root.priors[key] + eps * float(noise[i])


def _build_visit_distribution(
    root: MCTSNode, temperature: float
) -> tuple[dict[int, float], Move]:
    """由根节点访问次数构建 π 并采样走法。"""
    keys = list(root.children.keys())
    if not keys:
        # 无任何模拟完成（极端超时）：退化为对未物化走法的均匀分布
        if not root.untried_moves:
            raise ValueError("no legal moves available")
        probs = np.full(len(root.untried_moves), 1.0 / len(root.untried_moves))
        pi = {
            move_key(m): float(p)
            for m, p in zip(root.untried_moves, probs, strict=True)
        }
        return pi, root.untried_moves[0]

    visits = np.array([root.children[k].n_visits for k in keys], dtype=np.float64)
    total = visits.sum()
    if total <= 0.0:
        idx = int(np.random.randint(len(keys)))
    elif temperature == 0.0:
        idx = int(np.argmax(visits))
    else:
        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()
        idx = int(np.random.choice(len(keys), p=probs))

    pi_arr = visits / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))
    best_child = root.children[keys[idx]]
    assert best_child.move is not None
    return {k: float(p) for k, p in zip(keys, pi_arr, strict=True)}, best_child.move


def _timed_out(start_time: float, time_budget_ms: float) -> bool:
    """是否超时（预留 10% 安全边界）。time_budget_ms <= 0 表示不限时。"""
    if time_budget_ms <= 0:
        return False
    return (time.time() - start_time) * 1000.0 > time_budget_ms * 0.9


# =============================================================================
# 网络引导 MCTS
# =============================================================================


class MCTSEngine(Engine):
    """网络引导的 MCTS 搜索引擎（AlphaZero 风格）。"""

    def __init__(
        self,
        model: _PolicyValueNet,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        """初始化 MCTS 引擎。

        Args:
            model: 策略-价值网络（需实现 predict）
            num_simulations: 每次搜索的模拟次数
            c_puct: PUCT 探索常数
            temperature: 走子温度（0 = 贪心）
            dirichlet_alpha: Dirichlet 噪声集中度（self-play 根节点用）
            dirichlet_epsilon: Dirichlet 噪声权重
        """
        self._model: _PolicyValueNet = model
        self._num_simulations: int = num_simulations
        self._c_puct: float = c_puct
        self._temperature: float = temperature
        self._dirichlet_alpha: float = dirichlet_alpha
        self._dirichlet_epsilon: float = dirichlet_epsilon
        self._total_simulations: int = 0

    # ------------------------------------------------------------------ Engine

    def search(self, board: Board, time_budget_ms: float) -> Move:
        """执行 num_simulations 次模拟，返回访问次数最多的走法（temperature=0）。

        Args:
            board: 当前局面（不会被修改，内部 clone）
            time_budget_ms: 时间预算（ms），≤0 表示不限时

        Returns:
            最优走法（根节点子节点中访问次数最多者）
        """
        root = self._run_search(board, time_budget_ms, add_noise=False)
        _, move = self._visit_distribution(root, 0.0)
        return move

    def get_name(self) -> str:
        """返回引擎名称。"""
        return f"MCTS(simulations={self._num_simulations})"

    def reset(self) -> None:
        """重置累计统计。MCTS 每步重建搜索树，无需保留状态。"""
        self._total_simulations = 0

    # ----------------------------------------------------------- self-play 接口

    def get_policy(
        self, board: Board, temperature: float, add_noise: bool = False
    ) -> tuple[dict[int, float], Move]:
        """搜索后返回 (走法访问次数分布 π, 采样走法)。

        π 以 {move 键: 概率} 形式返回（已归一化的访问次数分布），供 selfplay
        记录训练标签；采样走法按温度 τ 从访问分布采样（τ=0 贪心）。

        Args:
            board: 当前局面
            temperature: 采样温度
            add_noise: 是否在根节点注入 Dirichlet 噪声（self-play 前 30 步）
        """
        root = self._run_search(board, 0.0, add_noise=add_noise)
        return self._visit_distribution(root, temperature)

    # ------------------------------------------------------------------ 核心

    def _run_search(self, board: Board, time_budget_ms: float, add_noise: bool) -> MCTSNode:
        root = _build_root(board)
        if root.is_terminal:
            raise ValueError("search() called on terminal position")

        start_time = time.time()
        self._expand(root)  # 网络评估根节点，得到先验
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self._num_simulations):
            if _timed_out(start_time, time_budget_ms):
                break
            self._simulate(root)
            self._total_simulations += 1
        return root

    def _simulate(self, root: MCTSNode) -> None:
        """单次模拟：选择 → 扩展/评估 → 回传（negamax 视角翻转）。"""
        node = root
        path: list[MCTSNode] = [root]

        while not node.is_terminal and node.priors and (node.children or node.untried_moves):
            node = self._select_child(node)
            path.append(node)

        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node)

        for n in reversed(path):
            n.n_visits += 1
            n.total_value += value
            value = -value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """PUCT 选择（委托给模块级实现，供批量 MCTS 复用）。"""
        return _select_child_puct(node, self._c_puct)

    def _expand(self, node: MCTSNode) -> float:
        """网络评估节点：计算先验并保存（懒展开，不物化子节点）。"""
        policy_logits, value = self._model.predict(node.board, node.untried_moves)
        node.priors = _compute_priors(node.untried_moves, policy_logits)
        return float(value)

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """在根节点先验上注入 Dirichlet 噪声（委托给模块级实现）。"""
        _inject_dirichlet_noise(root, self._dirichlet_alpha, self._dirichlet_epsilon)

    def _visit_distribution(
        self, root: MCTSNode, temperature: float
    ) -> tuple[dict[int, float], Move]:
        """由根节点访问次数构建 π 并采样走法（委托给模块级实现）。"""
        return _build_visit_distribution(root, temperature)


# =============================================================================
# 纯 MCTS 基线（无网络）
# =============================================================================


class PureMCTSEngine(Engine):
    """无网络引导的 MCTS 基线：UCB1 选择 + 随机模拟（rollout）评估。"""

    def __init__(self, num_simulations: int = 2000, c_puct: float = 1.0) -> None:
        """初始化纯 MCTS 引擎。

        Args:
            num_simulations: 每次搜索的模拟次数
            c_puct: UCB1 探索常数
        """
        self._num_simulations: int = num_simulations
        self._c_puct: float = c_puct

    def search(self, board: Board, time_budget_ms: float) -> Move:
        """执行模拟并返回访问次数最多的走法。"""
        root = _build_root(board)
        if root.is_terminal:
            raise ValueError("search() called on terminal position")

        start_time = time.time()
        for _ in range(self._num_simulations):
            if _timed_out(start_time, time_budget_ms):
                break
            self._simulate(root)

        keys = list(root.children.keys())
        if not keys:
            return root.untried_moves[0]
        best_key = keys[int(np.argmax([root.children[k].n_visits for k in keys]))]
        move = root.children[best_key].move
        assert move is not None
        return move

    def get_name(self) -> str:
        """返回引擎名称。"""
        return f"PureMCTS(simulations={self._num_simulations})"

    def reset(self) -> None:
        """纯 MCTS 无内部状态，无需重置。"""

    def _simulate(self, root: MCTSNode) -> None:
        """单次模拟：UCB1 选择 → 展开一个子节点 → rollout → 回传。"""
        node = root
        path: list[MCTSNode] = [root]

        while not node.untried_moves and not node.is_terminal:
            node = self._select_child(node)
            path.append(node)

        if node.is_terminal:
            value = node.terminal_value
        else:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            child = _make_child(node, move)
            node.children[move_key(move)] = child
            path.append(child)
            value = child.terminal_value if child.is_terminal else self._rollout_value(child.board)

        for n in reversed(path):
            n.n_visits += 1
            n.total_value += value
            value = -value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """UCB1 选择：Q + c * sqrt(ln N_parent / N_child)。"""
        ln_parent = math.log(max(node.n_visits, 1))
        best_child: MCTSNode | None = None
        best_score = -float("inf")

        for child in node.children.values():
            if child.n_visits == 0:
                score = float("inf")
            else:
                q_parent = -child.q_value()
                score = q_parent + self._c_puct * math.sqrt(ln_parent / child.n_visits)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("select_child() called on a childless node")
        return best_child

    @staticmethod
    def _rollout_value(board: Board) -> float:
        """从当前局面双方随机走子至终局，返回当前行棋方视角的 +1/-1。"""
        player = board.get_current_player()
        sim = board.clone()
        while not Rules.is_game_over(sim):
            moves = Rules.get_legal_moves(sim)
            if not moves:
                break
            sim.make_move(random.choice(moves))
        return 1.0 if Rules.get_winner(sim) == player else -1.0
