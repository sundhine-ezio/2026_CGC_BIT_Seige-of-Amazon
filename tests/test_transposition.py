"""ai/transposition.py 的单元测试。"""

import pytest
from src.ai.transposition import TranspositionEntry, TranspositionTable, ZobristHasher
from src.game.board import Board


class TestZobristHasher:
    """测试 Zobrist 哈希。"""

    def test_init_only_once(self) -> None:
        """第二次 init 不改变已有随机数。"""
        ZobristHasher.init(seed=42)
        h1: int = ZobristHasher.hash_piece(1, 0, 0)
        ZobristHasher.init(seed=999)  # 不应改变
        h2: int = ZobristHasher.hash_piece(1, 0, 0)
        assert h1 == h2

    def test_different_pieces_different_hash(self) -> None:
        """不同棋子的哈希值应不同。"""
        ZobristHasher.init(seed=42)
        h_white: int = ZobristHasher.hash_piece(1, 0, 0)
        h_black: int = ZobristHasher.hash_piece(-1, 0, 0)
        assert h_white != h_black

    def test_same_board_same_hash(self) -> None:
        """两个初始棋盘应产生相同哈希。"""
        b1: Board = Board()
        b2: Board = Board()
        assert b1.get_zobrist_hash() == b2.get_zobrist_hash()


class TestTranspositionTable:
    """测试置换表。"""

    @pytest.fixture
    def tt(self) -> TranspositionTable:
        """创建置换表。"""
        return TranspositionTable(size=1024)

    def test_store_and_lookup(self, tt: TranspositionTable) -> None:
        """存储后应能查找。"""
        tt.store(hash_key=12345, depth=3, value=0.75, flag=TranspositionTable.EXACT)
        entry: TranspositionEntry | None = tt.lookup(12345)
        assert entry is not None
        assert entry.value == 0.75
        assert entry.depth == 3
        assert entry.flag == TranspositionTable.EXACT

    def test_miss(self, tt: TranspositionTable) -> None:
        """未存储的哈希应返回 None。"""
        assert tt.lookup(99999) is None

    def test_depth_priority_replacement(self, tt: TranspositionTable) -> None:
        """深度优先替换: 更深覆盖更浅。"""
        # 同一个索引（冲突）
        # 使用能映射到同一槽位的两个不同 hash_key
        tt.store(hash_key=0, depth=2, value=0.5, flag=TranspositionTable.EXACT)
        tt.store(hash_key=1024, depth=4, value=0.8, flag=TranspositionTable.EXACT)  # 同索引

        # 原来的应该被覆盖
        assert tt.lookup(0) is None  # hash_key 0 被覆盖
        entry: TranspositionEntry | None = tt.lookup(1024)
        assert entry is not None
        assert entry.value == 0.8

    def test_shallow_does_not_replace_deep(self, tt: TranspositionTable) -> None:
        """浅搜索不覆盖深搜索结果。"""
        tt.store(hash_key=0, depth=5, value=0.9, flag=TranspositionTable.EXACT)
        tt.store(hash_key=0, depth=2, value=0.3, flag=TranspositionTable.EXACT)

        entry: TranspositionEntry | None = tt.lookup(0)
        assert entry is not None
        assert entry.depth == 5  # 保持旧的更深条目
        assert entry.value == 0.9

    def test_clear(self, tt: TranspositionTable) -> None:
        """清空后应查不到。"""
        tt.store(hash_key=42, depth=1, value=0.5, flag=TranspositionTable.EXACT)
        tt.clear()
        assert tt.lookup(42) is None
