"""开局库管理。

从 JSON 文件加载预设开局走法，支持按 Zobrist 哈希查询与加权随机选择。
"""

from __future__ import annotations

import json
import os
import random

from src.game.move import Move


class OpeningBook:
    """开局库——以 Zobrist 哈希值为键的走法字典。

    数据文件格式 (JSON):
        {
            "version": "1.0",
            "description": "...",
            "positions": {
                "<hash_hex>": [
                    {"move": "d10-d5-g7", "weight": 100, "comment": "..."},
                    ...
                ]
            }
        }
    """

    def __init__(self, book_path: str | None = None) -> None:
        """加载开局库 JSON 文件。

        Args:
            book_path: JSON 文件路径。若为 None 或文件不存在，开局库为空（禁用）。

        Raises:
            FileNotFoundError: 指定了路径但文件不存在
            ValueError: JSON 格式错误
        """
        self._positions: dict[int, list[dict[str, object]]] = {}

        if book_path is None or not os.path.exists(book_path):
            self._loaded: bool = False
            return

        try:
            with open(book_path, "r", encoding="utf-8") as f:
                data: dict[str, object] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"开局库 JSON 格式错误: {e}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"开局库文件不存在: {book_path}")

        positions_raw: dict[str, list[dict[str, object]]] = data.get("positions", {})  # type: ignore[assignment]
        for hex_key, moves in positions_raw.items():
            hash_key: int = int(hex_key, 16)
            self._positions[hash_key] = moves

        self._loaded: bool = True

    def query(self, board_hash: int) -> list[dict[str, object]] | None:
        """查询指定局面的开局走法候选。

        Args:
            board_hash: 64 位 Zobrist 哈希值

        Returns:
            候选走法列表（每项含 move/weight/comment），或 None（未命中）
        """
        if not self._loaded:
            return None
        return self._positions.get(board_hash)

    def get_random_move(self, board_hash: int) -> Move | None:
        """加权随机选取一个开局走法。

        Args:
            board_hash: 64 位 Zobrist 哈希值

        Returns:
            选中的 Move 对象，或 None（未命中或无可用权重＞0的走法）
        """
        candidates: list[dict[str, object]] | None = self.query(board_hash)
        if candidates is None or len(candidates) == 0:
            return None

        moves: list[str] = [str(c["move"]) for c in candidates]
        weights: list[float] = [float(c.get("weight", 1)) for c in candidates]

        total_weight: float = sum(weights)
        if total_weight <= 0:
            return None

        probabilities: list[float] = [w / total_weight for w in weights]

        chosen: str = random.choices(moves, weights=probabilities, k=1)[0]
        return Move.from_string(chosen)

    def is_available(self) -> bool:
        """开局库是否可用（已成功加载且非空）。"""
        return self._loaded and len(self._positions) > 0

    def get_total_positions(self) -> int:
        """返回开局库中的局面总数。"""
        return len(self._positions)
