"""游戏规则引擎。

提供合法走法生成、皇后走法可达范围计算、终局判断等纯函数。
"""

from __future__ import annotations

from src.game.board import Board
from src.game.constants import (
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
    def is_game_over(board: Board) -> bool:
        """判断当前局面是否终局。

        Returns:
            True 如果当前玩家无合法走法
        """
        return len(Rules.get_legal_moves(board)) == 0

    @staticmethod
    def get_winner(board: Board) -> int:
        """返回获胜方（应在 is_game_over 为 True 后调用）。

        Amazon 棋盘不存在平局，无合法走法的一方判负。

        Returns:
            WHITE(1) 或 BLACK(-1)
        """
        # 当前玩家无合法走法 → 对方胜
        return -board.get_current_player()
