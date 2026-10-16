"""棋盘状态表示。

Board 类负责管理 10×10 棋盘状态，支持走法的执行/撤销/深拷贝，
以及 Zobrist 哈希值的增量更新。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.game.constants import (
    BLACK,
    BOARD_SIZE,
    BURNED,
    EMPTY,
    INITIAL_BLACK_POSITIONS,
    INITIAL_WHITE_POSITIONS,
    WHITE,
)
from src.game.move import Move


@dataclass
class _UndoInfo:
    """走法撤销信息，存储在 move_history 栈中。"""

    from_pos: tuple[int, int]
    to_pos: tuple[int, int]
    arrow_pos: tuple[int, int]
    captured_from: int
    captured_to: int
    captured_arrow: int
    prev_hash: int
    prev_player: int


class Board:
    """10×10 亚马逊棋棋盘。

    内部状态:
        grid: (10,10) numpy 数组——EMPTY/WHITE/BLACK/BURNED
        current_player: 当前应走子方——WHITE(1) 或 BLACK(-1)
        amazon_positions: {WHITE: [(r,c),...], BLACK: [(r,c),...]}
        move_history: 走法历史栈（用于撤消）
        zobrist_hash: 64 位增量哈希值
    """

    def __init__(self) -> None:
        """初始化标准 10×10 棋盘，白方先行。"""
        self._grid: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # 放置黑方棋子
        for r, c in INITIAL_BLACK_POSITIONS:
            self._grid[r, c] = BLACK

        # 放置白方棋子
        for r, c in INITIAL_WHITE_POSITIONS:
            self._grid[r, c] = WHITE

        self._current_player: int = WHITE

        self._amazon_positions: dict[int, list[tuple[int, int]]] = {
            WHITE: list(INITIAL_WHITE_POSITIONS),
            BLACK: list(INITIAL_BLACK_POSITIONS),
        }

        self._move_history: list[_UndoInfo] = []

        # Zobrist 哈希将在 ai-transposition 模块完成后通过 _rehash() 计算
        # 先设为 0，后续由 TranspositionTable 集成时调用 _rehash()
        self._zobrist_hash: int = 0
        self._compute_initial_hash()

    # =========================================================================
    # 状态访问
    # =========================================================================

    def get_grid(self) -> np.ndarray:
        """返回棋盘矩阵的拷贝（防止外部修改）。

        Returns:
            (10, 10) numpy 数组
        """
        return self._grid.copy()

    def get_piece_at(self, row: int, col: int) -> int:
        """返回指定格子的内容。

        Args:
            row: 行号 (0-9)
            col: 列号 (0-9)

        Returns:
            EMPTY(0) / WHITE(1) / BLACK(-1) / BURNED(2)
        """
        return int(self._grid[row, col])

    def get_current_player(self) -> int:
        """返回当前应走子的一方。

        Returns:
            WHITE(1) 或 BLACK(-1)
        """
        return self._current_player

    def get_amazon_positions(self, player: int) -> list[tuple[int, int]]:
        """返回指定玩家所有亚马逊的位置（拷贝）。

        Args:
            player: WHITE 或 BLACK

        Returns:
            位置元组列表
        """
        return list(self._amazon_positions[player])

    def get_zobrist_hash(self) -> int:
        """返回当前局面的 Zobrist 哈希值。"""
        return self._zobrist_hash

    # =========================================================================
    # 走法执行与撤消
    # =========================================================================

    def make_move(self, move: Move) -> None:
        """在棋盘上执行一步走法（原位修改，不创建新棋盘）。

        执行顺序:
            1. 保存撤销信息
            2. 移动棋子（from→EMPTY, to→棋子）
            3. 射箭（arrow→BURNED）
            4. 更新棋子位置表
            5. 更新 Zobrist 哈希
            6. 切换玩家

        Args:
            move: 待执行的走法（调用者确保为合法走法）
        """
        piece: int = int(self._grid[move.from_pos])

        # 保存撤销信息
        undo: _UndoInfo = _UndoInfo(
            from_pos=move.from_pos,
            to_pos=move.to_pos,
            arrow_pos=move.arrow_pos,
            captured_from=piece,
            captured_to=int(self._grid[move.to_pos]),
            captured_arrow=int(self._grid[move.arrow_pos]),
            prev_hash=self._zobrist_hash,
            prev_player=self._current_player,
        )

        # 执行移动
        self._grid[move.from_pos] = EMPTY
        self._grid[move.to_pos] = piece
        self._grid[move.arrow_pos] = BURNED

        # 更新棋子位置表
        self._amazon_positions[piece].remove(move.from_pos)
        self._amazon_positions[piece].append(move.to_pos)

        # 更新 Zobrist 哈希
        self._zobrist_hash = self._update_hash_for_move(
            self._zobrist_hash, piece, move
        )

        # 切换玩家
        self._current_player = -self._current_player

        # 保存历史
        self._move_history.append(undo)

    def undo_move(self) -> None:
        """撤销最近一步走法。

        从 move_history 栈顶弹出撤销信息，恢复所有被修改的状态。

        Raises:
            IndexError: move_history 为空时调用
        """
        if not self._move_history:
            raise IndexError("undo_move() called with empty move_history")

        undo: _UndoInfo = self._move_history.pop()

        # 恢复棋子
        self._grid[undo.from_pos] = undo.captured_from
        self._grid[undo.to_pos] = undo.captured_to
        self._grid[undo.arrow_pos] = undo.captured_arrow

        # 恢复棋子位置表
        piece: int = undo.captured_from
        self._amazon_positions[piece].remove(undo.to_pos)
        self._amazon_positions[piece].append(undo.from_pos)

        # 恢复哈希和玩家
        self._zobrist_hash = undo.prev_hash
        self._current_player = undo.prev_player

    def clone(self) -> Board:
        """深拷贝当前棋盘。

        用于蒙特卡洛评估中的随机模拟，避免影响原始搜索状态。

        Returns:
            与当前棋盘状态完全相同的新 Board 对象
        """
        new_board: Board = Board.__new__(Board)

        new_board._grid = self._grid.copy()
        new_board._current_player = self._current_player

        new_board._amazon_positions = {
            WHITE: list(self._amazon_positions[WHITE]),
            BLACK: list(self._amazon_positions[BLACK]),
        }

        # 拷贝 move_history（undo_info 中的元组和基本类型无需深拷贝）
        new_board._move_history = list(self._move_history)

        new_board._zobrist_hash = self._zobrist_hash

        return new_board

    def reset(self) -> None:
        """重置棋盘到标准初始布局。"""
        self._grid.fill(EMPTY)
        for r, c in INITIAL_BLACK_POSITIONS:
            self._grid[r, c] = BLACK
        for r, c in INITIAL_WHITE_POSITIONS:
            self._grid[r, c] = WHITE

        self._current_player = WHITE
        self._amazon_positions = {
            WHITE: list(INITIAL_WHITE_POSITIONS),
            BLACK: list(INITIAL_BLACK_POSITIONS),
        }
        self._move_history = []
        self._compute_initial_hash()

    # =========================================================================
    # Zobrist 哈希（内部实现；ai-transposition 模块提供哈希函数）
    # =========================================================================

    def _compute_initial_hash(self) -> None:
        """计算初始局面的 Zobrist 哈希值。

        若 ZobristHasher 已初始化，使用增量 XOR 计算；
        否则设为 0（待后续集成）。
        """
        try:
            from src.ai.transposition import ZobristHasher  # noqa: F811

            h: int = ZobristHasher.hash_side_to_move()
            for r, c in INITIAL_BLACK_POSITIONS:
                h ^= ZobristHasher.hash_piece(BLACK, r, c)
            for r, c in INITIAL_WHITE_POSITIONS:
                h ^= ZobristHasher.hash_piece(WHITE, r, c)
            self._zobrist_hash = h
        except ImportError:
            self._zobrist_hash = 0

    @staticmethod
    def _update_hash_for_move(
        current_hash: int, piece: int, move: Move
    ) -> int:
        """增量更新 Zobrist 哈希值（执行走法后调用）。

        注意: 此方法在 make_move 中调用，使用延迟导入避免循环依赖。
        """
        try:
            from src.ai.transposition import ZobristHasher

            h: int = current_hash
            # 移除 from_pos 的棋子
            h ^= ZobristHasher.hash_piece(piece, *move.from_pos)
            # 放置 to_pos 的棋子
            h ^= ZobristHasher.hash_piece(piece, *move.to_pos)
            # 放置烧毁格
            h ^= ZobristHasher.hash_piece(BURNED, *move.arrow_pos)
            # 切换走子方
            h ^= ZobristHasher.hash_side_to_move()
            return h
        except ImportError:
            return current_hash
