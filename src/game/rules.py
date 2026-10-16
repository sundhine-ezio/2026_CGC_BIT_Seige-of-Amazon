"""游戏规则引擎。

提供合法走法生成、皇后走法可达范围计算、终局判断等纯函数。
"""

from __future__ import annotations

import numpy as np

from src.game.board import Board
from src.game.constants import (
    BOARD_SIZE,
    EMPTY,
    QUEEN_DIRECTIONS,
)
from src.game.move import Move


class Rules:
    """亚马逊棋规则引擎。所有方法为静态方法，不持有状态。"""

    @staticmethod
    def get_queen_reachable(
        board: Board, from_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """返回从 from_pos 出发，皇后走法可达的所有空格位置。

        沿 8 个方向步进，遇非空格（棋子或烧毁格）或边界停止该方向。

        Args:
            board: 当前棋盘状态
            from_pos: 出发点 (row, col)

        Returns:
            可达空格位置列表（各方向均含 from_pos 本身之外的空格）
        """
        from_r: int
        from_c: int
        from_r, from_c = from_pos

        reachable: list[tuple[int, int]] = []

        for dr, dc in QUEEN_DIRECTIONS:
            r: int = from_r + dr
            c: int = from_c + dc
            while 0 <= r < 10 and 0 <= c < 10:
                if board.get_piece_at(r, c) == EMPTY:
                    reachable.append((r, c))
                    r += dr
                    c += dc
                else:
                    break  # 遇到非空格，停止该方向

        return reachable

    @staticmethod
    def get_legal_moves(board: Board) -> list[Move]:
        """生成当前局面下 current_player 的所有合法走法。

        对当前玩家的每个亚马逊：
          1. 生成其皇后走法可达的所有空格 → 移动候选
          2. 对每个移动候选，模拟移动后从新位置生成箭落点候选
          3. 每个 (起点, 移动终点, 箭落点) 为一个合法走法

        Args:
            board: 当前棋盘状态

        Returns:
            合法走法列表。若无合法走法返回空列表。
        """
        player: int = board.get_current_player()
        amazons: list[tuple[int, int]] = board.get_amazon_positions(player)

        moves: list[Move] = []

        for amazon_pos in amazons:
            # 第一步：该亚马逊的合法移动目标
            move_targets: list[tuple[int, int]] = Rules.get_queen_reachable(
                board, amazon_pos
            )

            for move_target in move_targets:
                # 模拟移动（用 make_move + undo_move 保证正确性）
                temp_move: Move = Move(
                    from_pos=amazon_pos,
                    to_pos=move_target,
                    arrow_pos=move_target,  # 箭位置暂设为移动目标（后续不会用到）
                )
                board.make_move(temp_move)

                # 第二步：从移动终点生成箭的合法落点
                # 注意：箭可以从新位置出发，落点可以是原来的 from_pos（现在为空）
                arrow_targets: list[tuple[int, int]] = Rules.get_queen_reachable(
                    board, move_target
                )

                # 恢复棋子位置
                board.undo_move()

                # 生成完整走法
                for arrow_target in arrow_targets:
                    moves.append(Move(amazon_pos, move_target, arrow_target))

        return moves

    @staticmethod
    def get_legal_arrow_targets(
        board: Board, from_pos: tuple[int, int], to_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """返回亚马逊从 from_pos 走到 to_pos 之后，所有合法的箭落点。

        关键：本方法在**走子后**的局面上计算，因此亚马逊刚腾出的 from_pos
        也是合法落点。直接用 ``get_queen_reachable(board, to_pos)`` 是在走子
        前算的，那时 from_pos 还被自己的棋子占着，会漏掉「把箭射回原位」这类
        常见走法。校验人类落箭必须用本方法，才能与 AI（走 get_legal_moves
        路径）的口径一致。

        Args:
            board: 走子前的棋盘状态（不会被修改）
            from_pos: 亚马逊起点
            to_pos: 亚马逊落点

        Returns:
            合法箭落点列表（不含 to_pos 自身，那里将被亚马逊占据）
        """
        probe: Board = board.clone()
        # arrow_pos=to_pos 作为占位：模拟「只移动、尚未射箭」，
        # Rules.get_legal_moves 内部用的是同一技巧。
        probe.make_move(Move(from_pos=from_pos, to_pos=to_pos, arrow_pos=to_pos))
        return Rules.get_queen_reachable(probe, to_pos)

    @staticmethod
    def has_any_legal_move(board: Board) -> bool:
        """快速判断当前玩家是否至少有一个合法走法（找到即返回）。

        与 ``len(get_legal_moves(board)) > 0`` 等价，但不构造任何 Move 对象、
        也不做 make_move/undo_move 模拟，因此快一两个数量级。终局判断只关心
        "有没有"，用本方法；需要走法列表时才用 get_legal_moves。

        Args:
            board: 当前棋盘状态

        Returns:
            True 如果当前玩家存在至少一个合法走法
        """
        grid: np.ndarray = board.get_grid()
        player: int = board.get_current_player()

        for from_r, from_c in board.get_amazon_positions(player):
            for dr, dc in QUEEN_DIRECTIONS:
                to_r, to_c = from_r + dr, from_c + dc
                # 沿该方向步进到每个可达空格
                while (
                    0 <= to_r < BOARD_SIZE
                    and 0 <= to_c < BOARD_SIZE
                    and grid[to_r, to_c] == EMPTY
                ):
                    if Rules._can_shoot_arrow(grid, to_r, to_c, from_r, from_c):
                        return True
                    to_r += dr
                    to_c += dc

        return False

    @staticmethod
    def _can_shoot_arrow(
        grid: np.ndarray,
        amazon_r: int,
        amazon_c: int,
        vacated_r: int,
        vacated_c: int,
    ) -> bool:
        """判断亚马逊位于 (amazon_r, amazon_c) 时能否射出至少一箭。

        箭必须落在空格上，且沿 8 方向的第一格若被非空格挡住则该方向无落点。
        因此只需检查相邻八格：任一为空格（或为本次移动腾出的空格）即可。

        Args:
            grid: 棋盘数组
            amazon_r: 亚马逊当前所在行
            amazon_c: 亚马逊当前所在列
            vacated_r: 本次移动腾出的空格行
            vacated_c: 本次移动腾出的空格列

        Returns:
            True 如果能射出至少一箭
        """
        for dr, dc in QUEEN_DIRECTIONS:
            r, c = amazon_r + dr, amazon_c + dc
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                continue
            # 亚马逊移动后原位置变为空格，可作落点
            if grid[r, c] == EMPTY or (r == vacated_r and c == vacated_c):
                return True
        return False

    @staticmethod
    def is_game_over(board: Board) -> bool:
        """判断当前局面是否终局。

        用 has_any_legal_move 早停检测，不做全量走法生成。

        Returns:
            True 如果当前玩家无合法走法
        """
        return not Rules.has_any_legal_move(board)

    @staticmethod
    def get_winner(board: Board) -> int:
        """返回获胜方（应在 is_game_over 为 True 后调用）。

        Amazon 棋盘不存在平局，无合法走法的一方判负。

        Returns:
            WHITE(1) 或 BLACK(-1)
        """
        # 当前玩家无合法走法 → 对方胜
        return -board.get_current_player()
