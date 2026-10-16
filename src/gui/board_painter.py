"""棋盘绘制组件。

Tkinter Canvas 子类，负责渲染 10×10 棋盘、棋子、烧毁格和走法高亮。
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from src.game.board import Board
from src.game.constants import BLACK, BOARD_SIZE, BURNED, WHITE
from src.game.move import Move


class BoardCanvas(tk.Canvas):
    """棋盘绘制画布。"""

    CELL_SIZE: int = 60
    PIECE_RADIUS: int = 22
    ARROW_RADIUS: int = 5
    BOARD_PADDING: int = 30
    BOARD_COLOR: str = "#DEB887"
    LINE_COLOR: str = "#000000"
    WHITE_COLOR: str = "#FFFFFF"
    BLACK_COLOR: str = "#333333"
    BURNED_COLOR: str = "#999999"
    HIGHLIGHT_COLOR: str = "#FFD700"

    def __init__(self, parent: tk.Widget) -> None:
        canvas_size: int = self.CELL_SIZE * BOARD_SIZE + 2 * self.BOARD_PADDING
        super().__init__(
            parent,
            width=canvas_size,
            height=canvas_size,
            bg=self.BOARD_COLOR,
            highlightthickness=0,
        )

        self._click_callback: Callable[[int, int], None] | None = None
        self._last_move: Move | None = None

        self.bind("<Button-1>", self._on_click)

    # =========================================================================
    # 公共接口
    # =========================================================================

    def set_click_callback(self, callback: Callable[[int, int], None]) -> None:
        """注册棋盘点击回调。

        Args:
            callback: 回调函数 callback(row, col)
        """
        self._click_callback = callback

    def draw_board(self, board: Board) -> None:
        """绘制完整棋盘。

        Args:
            board: 当前棋盘状态
        """
        self.delete("all")
        self._draw_grid()
        self._draw_labels()

        grid = board.get_grid()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val: int = int(grid[r, c])
                if val == WHITE:
                    self._draw_piece(r, c, WHITE)
                elif val == BLACK:
                    self._draw_piece(r, c, BLACK)
                elif val == BURNED:
                    self._draw_burned(r, c)

        if self._last_move is not None:
            self._highlight_last_move(self._last_move)

    def set_last_move(self, move: Move) -> None:
        """设置最新走法（用于高亮）。"""
        self._last_move = move

    # =========================================================================
    # 内部绘制方法
    # =========================================================================

    def _grid_coords(self, row: int, col: int) -> tuple[int, int]:
        """棋盘坐标 → 格子中心像素坐标。"""
        x: int = self.BOARD_PADDING + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y: int = self.BOARD_PADDING + row * self.CELL_SIZE + self.CELL_SIZE // 2
        return (x, y)

    def _draw_grid(self) -> None:
        """绘制 10×10 网格线。"""
        padding: int = self.BOARD_PADDING
        size: int = self.CELL_SIZE * BOARD_SIZE
        for i in range(BOARD_SIZE + 1):
            pos: int = padding + i * self.CELL_SIZE
            # 水平线
            self.create_line(padding, pos, padding + size, pos, fill=self.LINE_COLOR)
            # 竖直线
            self.create_line(pos, padding, pos, padding + size, fill=self.LINE_COLOR)

    def _draw_labels(self) -> None:
        """绘制行列标注。"""
        padding: int = self.BOARD_PADDING
        size: int = self.CELL_SIZE * BOARD_SIZE

        # 列标 a-j（顶部）
        cols: str = "abcdefghij"
        for c in range(BOARD_SIZE):
            x: int = padding + c * self.CELL_SIZE + self.CELL_SIZE // 2
            self.create_text(x, padding // 2, text=cols[c], font=("Arial", 12))

        # 行标 10-1（右侧）
        for r in range(BOARD_SIZE):
            y: int = padding + r * self.CELL_SIZE + self.CELL_SIZE // 2
            label: str = str(BOARD_SIZE - r)
            self.create_text(padding + size + padding // 2, y, text=label, font=("Arial", 12))

    def _draw_piece(self, row: int, col: int, player: int) -> None:
        """绘制一枚棋子。

        Args:
            row, col: 棋盘坐标
            player: WHITE 或 BLACK
        """
        x, y = self._grid_coords(row, col)
        r: int = self.PIECE_RADIUS
        fill: str = self.WHITE_COLOR if player == WHITE else self.BLACK_COLOR
        outline: str = self.BLACK_COLOR if player == WHITE else self.WHITE_COLOR

        self.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)

    def _draw_burned(self, row: int, col: int) -> None:
        """绘制烧毁格标记。"""
        x, y = self._grid_coords(row, col)
        cell_left: int = x - self.CELL_SIZE // 2
        cell_top: int = y - self.CELL_SIZE // 2
        cell_right: int = cell_left + self.CELL_SIZE
        cell_bottom: int = cell_top + self.CELL_SIZE

        self.create_rectangle(
            cell_left, cell_top, cell_right, cell_bottom,
            fill=self.BURNED_COLOR, outline="",
        )
        # 绘制 × 标记
        margin: int = 8
        self.create_line(
            cell_left + margin, cell_top + margin,
            cell_right - margin, cell_bottom - margin,
            fill="#cc0000", width=2,
        )
        self.create_line(
            cell_right - margin, cell_top + margin,
            cell_left + margin, cell_bottom - margin,
            fill="#cc0000", width=2,
        )

    def _highlight_last_move(self, move: Move) -> None:
        """高亮最近一步走法的移动路径和箭落点。"""
        # 从 from_pos 到 to_pos 画箭头
        x1, y1 = self._grid_coords(*move.from_pos)
        x2, y2 = self._grid_coords(*move.to_pos)
        self.create_line(
            x1, y1, x2, y2, fill=self.HIGHLIGHT_COLOR, width=3, arrow=tk.LAST,
        )

        # 箭落点用高亮边框
        ax, ay = self._grid_coords(*move.arrow_pos)
        r: int = self.PIECE_RADIUS + 4
        self.create_oval(
            ax - r, ay - r, ax + r, ay + r,
            outline=self.HIGHLIGHT_COLOR, width=3,
        )

    def _pixel_to_board(self, x: int, y: int) -> tuple[int, int] | None:
        """像素坐标 → 棋盘 (row, col) 坐标。

        Returns:
            (row, col) 或 None（点击在棋盘外）
        """
        padding: int = self.BOARD_PADDING
        col: int = (x - padding) // self.CELL_SIZE
        row: int = (y - padding) // self.CELL_SIZE

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)
        return None

    def _on_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """鼠标点击事件处理。"""
        pos: tuple[int, int] | None = self._pixel_to_board(event.x, event.y)
        if pos is not None and self._click_callback is not None:
            self._click_callback(*pos)
