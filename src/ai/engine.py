"""AI 搜索引擎抽象基类。

所有搜索引擎（Alpha-Beta、MCTS 等）均需实现此接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.game.board import Board
from src.game.move import Move


class Engine(ABC):
    """搜索引擎抽象基类。"""

    @abstractmethod
    def search(self, board: Board, time_budget_ms: float) -> Move:
        """在给定棋盘状态和时间预算内，搜索并返回最优走法。

        Args:
            board: 当前棋盘状态（传入前应 clone，引擎不会修改原始对象）
            time_budget_ms: 可用时间预算（毫秒）。≤0 表示无限制。

        Returns:
            搜索到的最优走法。即使超时也必须返回合法走法。
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """返回引擎名称，用于日志和评估。

        Returns:
            引擎名称字符串
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置引擎内部状态（置换表、统计等），新一局开始时调用。"""
        ...
