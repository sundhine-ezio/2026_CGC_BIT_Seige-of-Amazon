"""局面评估器。

提供两种评估方式：
  - MonteCarloEvaluator: 蒙特卡洛随机模拟评估（精确但慢）
  - HeuristicEvaluator: 领地控制启发式评估（快速，适合深度搜索）
"""

from __future__ import annotations

import random

from src.game.board import Board
from src.game.constants import WIN
from src.game.rules import Rules


# =============================================================================
# 蒙特卡洛评估器（保留作为备选方案）
# =============================================================================


def _random_playout(board: Board) -> int:
    """从当前局面出发，双方随机走子至终局。

    Args:
        board: 棋盘状态（会被修改！调用者传入 clone）

    Returns:
        获胜方: WHITE(1) 或 BLACK(-1)
    """
    while not Rules.is_game_over(board):
        moves: list = Rules.get_legal_moves(board)
        if not moves:
            break
        move = random.choice(moves)
        board.make_move(move)

    return Rules.get_winner(board)


class MonteCarloEvaluator:
    """蒙特卡洛随机模拟局面评估器。

    每次评估执行若干次随机对弈模拟，以当前玩家获胜频率作为评估值。
    """

    def __init__(self, num_simulations: int = 100) -> None:
        """初始化评估器。

        Args:
            num_simulations: 每次 evaluate() 执行的随机模拟局数
        """
        self._num_simulations: int = num_simulations

    def evaluate(self, board: Board) -> float:
        """从当前局面出发，执行随机模拟并返回当前玩家的估计胜率。

        Args:
            board: 待评估的局面（不会被修改，内部使用 clone）

        Returns:
            float, 范围 [0.0, 1.0]——当前玩家视角的胜率估计
        """
        player: int = board.get_current_player()
        wins: int = 0

        for _ in range(self._num_simulations):
            sim_board: Board = board.clone()
            result: int = _random_playout(sim_board)
            if result == player:
                wins += 1

        return wins / self._num_simulations

    def set_num_simulations(self, n: int) -> None:
        """动态调整每次评估的模拟次数。

        Args:
            n: 新的模拟次数
        """
        self._num_simulations = n

    def get_num_simulations(self) -> int:
        """返回当前模拟次数设置。"""
        return self._num_simulations


# =============================================================================
# 启发式评估器（领地控制法——快速、确定性、无噪声）
# =============================================================================


class HeuristicEvaluator:
    """基于领地控制的快速启发式局面评估器。

    评估原理：
      对当前玩家和对手分别计算"领地"——所有己方亚马逊通过皇后走法
      能到达的独立格子总数。领地越大代表机动性越强、局面越好。
      最终得分 = 己方领地 / (己方领地 + 对方领地)，范围 [0, 1]。

    特点：
      - 确定性：同一局面多次评估结果完全相同
      - 极快：单次评估仅需遍历 8 枚棋子 × 8 方向 ≈ 数百次格子检查
      - 无噪声：不会像蒙特卡洛那样引入随机误差，alpha-beta 剪枝效率更高
    """

    def __init__(self) -> None:
        """初始化启发式评估器（无状态）。"""
        pass

    def evaluate(self, board: Board) -> float:
        """从当前玩家视角评估局面。

        计算当前玩家与对手的领地比。

        Args:
            board: 待评估的局面（不会被修改）

        Returns:
            float, 范围 [0.0, 1.0]:
              0.0  = 当前玩家无合法走法（必败）
              1.0  = 对手无合法走法（必胜）
              0.5  = 双方领地相当
              >0.5 = 当前玩家优势
        """
        player: int = board.get_current_player()
        opponent: int = -player

        my_territory: int = self._count_territory(board, player)
        opp_territory: int = self._count_territory(board, opponent)

        if my_territory == 0:
            return 0.0
        if opp_territory == 0:
            return 1.0

        return my_territory / (my_territory + opp_territory)

    @staticmethod
    def _count_territory(board: Board, player: int) -> int:
        """计算某方所有亚马逊能到达的独立格子总数。

        对玩家的每枚亚马逊，通过皇后走法计算可达空格，
        所有亚马逊的可达格子取并集（去重），返回总数。

        Args:
            board: 棋盘状态
            player: 玩家（WHITE 或 BLACK）

        Returns:
            独立可达格子总数
        """
        reachable: set[tuple[int, int]] = set()
        for pos in board.get_amazon_positions(player):
            for r, c in Rules.get_queen_reachable(board, pos):
                reachable.add((r, c))
        return len(reachable)

    # 保持与 MonteCarloEvaluator 接口兼容
    def set_num_simulations(self, n: int) -> None:
        """兼容接口（启发式评估不需要模拟次数）。"""
        pass

    def get_num_simulations(self) -> int:
        """兼容接口。"""
        return 0
