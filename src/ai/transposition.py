"""Zobrist 哈希与置换表。

ZobristHasher: 使用固定种子生成 64 位随机数表，支持增量哈希更新。
TranspositionTable: 用于 Alpha-Beta 搜索的置换表，深度优先替换策略。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.game.constants import BLACK, BOARD_SIZE, BURNED, WHITE
from src.game.move import Move


class ZobristHasher:
    """Zobrist 哈希生成器。随机数表在首次使用时以固定种子初始化。"""

    _table: dict[tuple[int, int, int], int] = {}
    _side_to_move_rand: int = 0
    _initialized: bool = False

    @classmethod
    def init(cls, seed: int = 42) -> None:
        """使用固定种子生成所有 Zobrist 随机数。

        对每种棋子类型 × 每个格子生成独立 64 位随机数，
        外加一个侧行权随机数。

        Args:
            seed: 随机种子（固定以保证可复现）
        """
        if cls._initialized:
            return

        rng: random.Random = random.Random(seed)

        # 生成 (piece, row, col) → 64-bit 随机数
        piece_types: list[int] = [WHITE, BLACK, BURNED]
        for piece in piece_types:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    # 生成 64 位随机数
                    cls._table[(piece, row, col)] = rng.getrandbits(64)

        # 当前走子方
        cls._side_to_move_rand = rng.getrandbits(64)

        cls._initialized = True

    @classmethod
    def hash_piece(cls, piece: int, row: int, col: int) -> int:
        """返回 (piece, row, col) 对应的随机哈希值。

        Args:
            piece: 棋子类型（WHITE/BLACK/BURNED）
            row: 行号
            col: 列号

        Returns:
            64 位随机数
        """
        if not cls._initialized:
            cls.init()
        return cls._table[(piece, row, col)]

    @classmethod
    def hash_side_to_move(cls) -> int:
        """返回当前走子方的哈希分量。"""
        if not cls._initialized:
            cls.init()
        return cls._side_to_move_rand


@dataclass
class TranspositionEntry:
    """置换表条目。"""

    hash_key: int           # 64 位完整哈希（用于冲突检测）
    depth: int              # 该条目记录的搜索深度
    value: float            # 评估值 [0, 1]
    flag: int               # EXACT=0 / LOWER_BOUND=1 / UPPER_BOUND=2
    best_move: Move | None  # 该局面下的最佳走法


class TranspositionTable:
    """置换表——使用数组 + 取模索引，深度优先替换。"""

    EXACT: int = 0
    LOWER_BOUND: int = 1
    UPPER_BOUND: int = 2

    def __init__(self, size: int = 2**20) -> None:
        """初始化置换表。

        Args:
            size: 表容量（应为 2 的幂以支持位运算取模），默认 2^20 ≈ 100 万
        """
        self._size: int = size
        self._table: list[TranspositionEntry | None] = [None] * size
        self._mask: int = size - 1  # size 为 2 的幂时，hash & mask == hash % size

    def _index(self, hash_key: int) -> int:
        """将哈希值映射到表索引。"""
        return hash_key & self._mask

    def lookup(self, hash_key: int) -> TranspositionEntry | None:
        """查询哈希值对应的置换表条目。

        Args:
            hash_key: 64 位 Zobrist 哈希值

        Returns:
            TranspositionEntry 若命中且完整哈希匹配，否则 None
        """
        idx: int = self._index(hash_key)
        entry: TranspositionEntry | None = self._table[idx]
        if entry is not None and entry.hash_key == hash_key:
            return entry
        return None

    def store(
        self,
        hash_key: int,
        depth: int,
        value: float,
        flag: int,
        best_move: Move | None = None,
    ) -> None:
        """存入置换表。

        替换策略: 槽位为空 或 新深度 ≥ 旧深度 时覆盖。

        Args:
            hash_key: 64 位完整哈希值
            depth: 搜索深度
            value: 评估值 [0, 1]
            flag: EXACT / LOWER_BOUND / UPPER_BOUND
            best_move: 最优走法（可选）
        """
        idx: int = self._index(hash_key)
        existing: TranspositionEntry | None = self._table[idx]

        if existing is None or depth >= existing.depth:
            self._table[idx] = TranspositionEntry(
                hash_key=hash_key,
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move,
            )

    def clear(self) -> None:
        """清空置换表。"""
        self._table = [None] * self._size
