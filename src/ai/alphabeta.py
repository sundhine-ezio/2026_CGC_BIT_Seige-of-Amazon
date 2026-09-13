"""Alpha-Beta 剪枝搜索引擎。

实现带迭代加深、置换表、走法排序、可插拔开局库/残局库的 Alpha-Beta 搜索。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.ai.engine import Engine
from src.ai.transposition import TranspositionEntry, TranspositionTable
from src.game.board import Board
from src.game.constants import DRAW
from src.game.move import Move
from src.game.rules import Rules

if TYPE_CHECKING:
    from src.opening_book.book import OpeningBook
    from src.endgame.tablebase import Tablebase


class _Evaluator(Protocol):
    """评估器接口：任何具有 evaluate(board) -> float 的对象均可作为评估器。"""

    def evaluate(self, board: Board) -> float:
        """评估局面，返回当前玩家视角的胜率 [0, 1]."""
        ...


class TimeoutError(Exception):
    """搜索超时异常。在迭代加深中被捕获以返回当前最优解。"""
    pass


@dataclass
class SearchResult:
    """一次搜索的返回值。"""

    move: Move | None
    value: float


@dataclass
class SearchStats:
    """搜索统计信息。"""

    nodes_visited: int = 0
    leaves_evaluated: int = 0
    transposition_hits: int = 0
    depth_reached: int = 0
    time_elapsed_ms: float = 0.0


class AlphaBetaEngine(Engine):
    """Alpha-Beta 搜索引擎。

    支持：
    - 迭代加深搜索（逐层加深，超时返回）
    - 置换表（避免重复搜索）
    - 启发式 / 蒙特卡洛叶节点评估
    - 走法排序（大幅提升剪枝效率）
    - 可选开局库（命中时直接返回）
    - 可选残局库（命中时返回精确值）
    """

    def __init__(
        self,
        evaluator: _Evaluator,
        max_depth: int = 5,
        time_budget_ms: float = 15000.0,
        use_transposition_table: bool = True,
        use_iterative_deepening: bool = True,
        use_move_ordering: bool = True,
        transposition_table_size: int = 2**20,
    ) -> None:
        """初始化 Alpha-Beta 引擎。

        Args:
            evaluator: 局面评估器（HeuristicEvaluator 或 MonteCarloEvaluator）
            max_depth: 最大搜索深度
            time_budget_ms: 单步默认时间预算（毫秒）
            use_transposition_table: 是否启用置换表
            use_iterative_deepening: 是否启用迭代加深
            use_move_ordering: 是否启用走法排序（默认开启，强烈建议启用）
            transposition_table_size: 置换表容量
        """
        self._evaluator: _Evaluator = evaluator
        self._max_depth: int = max_depth
        self._time_budget_ms: float = time_budget_ms
        self._use_transposition_table: bool = use_transposition_table
        self._use_iterative_deepening: bool = use_iterative_deepening
        self._use_move_ordering: bool = use_move_ordering

        self._transposition_table: TranspositionTable | None = (
            TranspositionTable(transposition_table_size)
            if use_transposition_table
            else None
        )

        self._opening_book: OpeningBook | None = None
        self._tablebase: Tablebase | None = None

        self._stats: SearchStats = SearchStats()

        # 超时标志（在搜索中由 _check_timeout 设置）
        self._timed_out: bool = False

    # =========================================================================
    # Engine 接口
    # =========================================================================

    def search(self, board: Board, time_budget_ms: float) -> Move:
        """搜索最优走法。

        流程:
            1. 保底: 取第一个合法走法
            2. 开局库: 命中则直接返回
            3. 迭代加深（或固定深度）搜索

        Args:
            board: 棋盘状态
            time_budget_ms: 时间预算（毫秒），≤0 表示无限制

        Returns:
            搜索到的最优走法
        """
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        if not legal_moves:
            raise ValueError("search() called on terminal position")

        fallback_move: Move = legal_moves[0]

        # 查询开局库
        if self._opening_book is not None and self._opening_book.is_available():
            book_move: Move | None = self._opening_book.get_random_move(
                board.get_zobrist_hash()
            )
            if book_move is not None:
                return book_move

        # 重置统计
        self._stats = SearchStats()
        self._timed_out = False
        start_time: float = time.time()

        search_board: Board = board.clone()
        effective_time: float = (
            time_budget_ms if time_budget_ms > 0 else float("inf")
        )

        result: SearchResult

        if self._use_iterative_deepening:
            result = self._iterative_deepening_search(
                search_board, start_time, effective_time
            )
        else:
            result = self._fixed_depth_search(
                search_board, self._max_depth, start_time, effective_time
            )

        self._stats.time_elapsed_ms = (time.time() - start_time) * 1000.0

        if result.move is None:
            return fallback_move
        return result.move

    def get_name(self) -> str:
        """返回引擎名称。"""
        return f"AlphaBeta(depth={self._max_depth})"

    def reset(self) -> None:
        """重置引擎内部状态。"""
        if self._transposition_table is not None:
            self._transposition_table.clear()
        self._stats = SearchStats()
        self._timed_out = False

    # =========================================================================
    # 搜索核心
    # =========================================================================

    def _iterative_deepening_search(
        self, board: Board, start_time: float, time_budget_ms: float
    ) -> SearchResult:
        """迭代加深搜索：从 depth=1 逐层加深至 max_depth。

        Args:
            board: 棋盘状态（会被修改，但退出时恢复）
            start_time: 搜索开始时间戳
            time_budget_ms: 时间预算

        Returns:
            当前搜索到的最优结果
        """
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        fallback_move: Move = legal_moves[0]
        best_result: SearchResult = SearchResult(move=fallback_move, value=0.0)

        for depth in range(1, self._max_depth + 1):
            if self._check_timeout(start_time, time_budget_ms):
                break

            # 每层搜索前重置超时标志
            self._timed_out = False
            result: SearchResult = self._fixed_depth_search(
                board.clone(), depth, start_time, time_budget_ms
            )

            if not self._timed_out:
                best_result = result
                self._stats.depth_reached = depth

            # 必胜 → 提前退出
            if best_result.value > 0.99:
                break

        return best_result

    def _fixed_depth_search(
        self,
        board: Board,
        depth: int,
        start_time: float,
        time_budget_ms: float,
    ) -> SearchResult:
        """固定深度搜索——对根节点的每个合法走法逐一评估。

        使用 negamax 模式：_alpha_beta 始终从 current_player 视角
        返回该玩家的最优胜率。根节点走子后调用 _alpha_beta（对方视角），
        再通过 1.0 - value 翻转为根玩家视角，选出最大值。

        Args:
            board: 棋盘状态（会被修改）
            depth: 目标搜索深度
            start_time: 开始时间戳
            time_budget_ms: 时间预算

        Returns:
            最优走法及评估值
        """
        legal_moves: list[Move] = Rules.get_legal_moves(board)
        if not legal_moves:
            return SearchResult(move=None, value=0.0)

        # 走法排序：优选走法在前，提高剪枝效率
        if self._use_move_ordering:
            legal_moves = self._order_moves(board, legal_moves)

        best_move: Move = legal_moves[0]
        best_value: float = -float("inf")

        for move in legal_moves:
            if self._check_timeout(start_time, time_budget_ms):
                break

            board.make_move(move)
            try:
                # 根玩家走子 → 轮到对方。_alpha_beta 返回对方最优胜率
                opponent_best: float = self._alpha_beta(
                    board,
                    depth - 1,
                    -float("inf"),
                    float("inf"),
                    start_time,
                    time_budget_ms,
                )
                # 翻转为根玩家视角
                value: float = 1.0 - opponent_best
            except TimeoutError:
                board.undo_move()
                self._timed_out = True
                break

            board.undo_move()

            if value > best_value:
                best_value = value
                best_move = move

        return SearchResult(move=best_move, value=best_value)

    def _alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
        time_budget_ms: float,
    ) -> float:
        """Negamax + Alpha-Beta 递归搜索核心。

        每一层都从 current_player 视角出发，选出对自己最优的走法。
        子节点返回值（对方视角）通过 1.0 - child_value 翻转。

        Args:
            board: 当前棋盘状态（会被 make/undo 修改）
            depth: 剩余搜索深度（0 = 叶节点）
            alpha: 当前玩家已知最优下界 [0, 1]
            beta:  当前玩家已知最优上界 [0, 1]
            start_time: 搜索开始时间戳
            time_budget_ms: 时间预算

        Returns:
            当前玩家视角的最优胜率 [0, 1]

        Raises:
            TimeoutError: 超时
        """
        # --- 终止条件 ---

        if self._check_timeout(start_time, time_budget_ms):
            raise TimeoutError()

        current_player: int = board.get_current_player()

        # 终局
        if Rules.is_game_over(board):
            winner: int = Rules.get_winner(board)
            return 1.0 if winner == current_player else 0.0

        # 深度耗尽 → 叶节点评估
        if depth == 0:
            self._stats.leaves_evaluated += 1

            # 查询置换表
            if self._transposition_table is not None:
                tt_entry: TranspositionEntry | None = (
                    self._transposition_table.lookup(board.get_zobrist_hash())
                )
                if tt_entry is not None and tt_entry.depth >= 0:
                    self._stats.transposition_hits += 1
                    return tt_entry.value

            # 查询残局库
            if self._tablebase is not None and self._tablebase.is_available():
                tb_result = self._tablebase.query(board.get_zobrist_hash())
                if tb_result is not None:
                    from src.game.constants import WIN as TB_WIN
                    return 1.0 if tb_result.result == TB_WIN else 0.0

            # 蒙特卡洛评估（返回 current_player 视角胜率）
            return self._evaluator.evaluate(board)

        # --- 走法生成 ---
        moves: list[Move] = Rules.get_legal_moves(board)
        if not moves:
            return 0.0  # 无合法走法 → 当前玩家负

        self._stats.nodes_visited += 1

        # --- Negamax 搜索：每层都为当前玩家选最大值 ---
        alpha_orig: float = alpha
        value: float = -float("inf")
        for move in moves:
            board.make_move(move)
            try:
                # 子节点 = 对方视角 → 翻转为己方视角
                # alpha/beta 边界也需要翻转并交换
                child_val: float = 1.0 - self._alpha_beta(
                    board,
                    depth - 1,
                    1.0 - beta,   # 己方下界 → 对方上界
                    1.0 - alpha,  # 己方上界 → 对方下界
                    start_time,
                    time_budget_ms,
                )
            except TimeoutError:
                board.undo_move()
                raise
            board.undo_move()

            if child_val > value:
                value = child_val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break  # 剪枝：父节点不会选这条路了

        # 存入置换表
        if self._transposition_table is not None:
            flag: int = (
                TranspositionTable.EXACT
                if value > alpha_orig and value < beta
                else TranspositionTable.LOWER_BOUND
                if value >= beta
                else TranspositionTable.UPPER_BOUND
            )
            self._transposition_table.store(
                board.get_zobrist_hash(), depth, value, flag
            )

        return value

    # =========================================================================
    # 走法排序
    # =========================================================================

    def _order_moves(
        self, board: Board, moves: list[Move]
    ) -> list[Move]:
        """按启发式评估对走法排序，优选的走法排在前面。

        原理：
          对当前玩家而言，好的走法会让对方（走子后的当前玩家）
          处于劣势。因此走子后用评估器从对方视角评估，得分越低
          表示对方越不利 → 该走法对当前玩家越有利。

          排序按对方视角评估值升序（低分 = 对方不利 = 己方有利）。

        Args:
            board: 当前棋盘状态（走子前，会被临时修改并在退出前恢复）
            moves: 待排序的走法列表

        Returns:
            排序后的走法列表（优选走法在前）
        """
        if not moves:
            return moves

        scored: list[tuple[float, Move]] = []
        for move in moves:
            board.make_move(move)
            # 走子后 current_player = 对方，评估值 = 对方视角的领地比
            score: float = self._evaluator.evaluate(board)
            board.undo_move()
            scored.append((score, move))

        # 升序排列：对方评估值越低 → 对方越不利 → 对我方越有利
        scored.sort(key=lambda item: item[0])
        return [m for _, m in scored]

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _check_timeout(self, start_time: float, time_budget_ms: float) -> bool:
        """检查是否已超时（预留 20% 安全边界）。

        Returns:
            True 如果已超时
        """
        if time_budget_ms == float("inf"):
            return False
        elapsed_ms: float = (time.time() - start_time) * 1000.0
        return elapsed_ms > time_budget_ms * 0.8

    def get_stats(self) -> SearchStats:
        """返回最近一次搜索的统计信息。"""
        return self._stats

    def set_opening_book(self, book: OpeningBook) -> None:
        """注入开局库。

        Args:
            book: OpeningBook 实例
        """
        self._opening_book = book

    def set_endgame_tablebase(self, tablebase: Tablebase) -> None:
        """注入残局库。

        Args:
            tablebase: Tablebase 实例
        """
        self._tablebase = tablebase
