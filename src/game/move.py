"""走法数据结构与编解码。

定义不可变的 Move 数据类，提供竞赛标准走法字符串与内部坐标
之间的双向转换。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.game.constants import algebraic_to_pos, pos_to_algebraic


@dataclass(frozen=True)
class Move:
    """不可变走法对象，表示一步完整的亚马逊棋走法。

    Attributes:
        from_pos: 起点 (row, col)——移动前亚马逊所在位置
        to_pos: 移动终点 (row, col)——亚马逊移动后的位置
        arrow_pos: 射箭落点 (row, col)——箭烧毁的格子
    """

    from_pos: tuple[int, int]
    to_pos: tuple[int, int]
    arrow_pos: tuple[int, int]

    @staticmethod
    def from_string(s: str) -> Move:
        """从竞赛标准走法字符串构造 Move 对象。

        格式: "起点-移动终点-箭落点"，每段为代数记谱法坐标。
        例: "d10-d5-g7"

        Args:
            s: 走法字符串

        Returns:
            Move 对象

        Raises:
            ValueError: 字符串格式非法（段数不对、坐标非法）
        """
        parts: list[str] = s.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"走法字符串格式错误: '{s}'——应为三段（起点-移动终点-箭落点），"
                f"实际得到 {len(parts)} 段"
            )

        from_pos: tuple[int, int] = algebraic_to_pos(parts[0])
        to_pos: tuple[int, int] = algebraic_to_pos(parts[1])
        arrow_pos: tuple[int, int] = algebraic_to_pos(parts[2])

        return Move(from_pos=from_pos, to_pos=to_pos, arrow_pos=arrow_pos)

    def to_string(self) -> str:
        """转换为竞赛标准走法字符串。

        Returns:
            格式为 "起点-移动终点-箭落点" 的字符串，如 "d10-d5-g7"
        """
        return "-".join([
            pos_to_algebraic(*self.from_pos),
            pos_to_algebraic(*self.to_pos),
            pos_to_algebraic(*self.arrow_pos),
        ])

    def __repr__(self) -> str:
        """等价于 to_string()。"""
        return self.to_string()
