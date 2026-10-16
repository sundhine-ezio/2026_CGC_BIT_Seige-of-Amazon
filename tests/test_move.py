"""game/move.py 的单元测试。"""

import pytest
from src.game.move import Move


class TestMoveFromString:
    """测试 Move.from_string 方法。"""

    def test_standard_format(self) -> None:
        """测试标准格式解析。"""
        move: Move = Move.from_string("d10-d5-g7")
        assert move.from_pos == (0, 3)
        assert move.to_pos == (5, 3)
        assert move.arrow_pos == (3, 6)

    def test_bottom_row(self) -> None:
        """测试第 1 行（row 9）的坐标。"""
        move: Move = Move.from_string("a1-j10-a1")
        assert move.from_pos == (9, 0)
        assert move.to_pos == (0, 9)
        assert move.arrow_pos == (9, 0)


class TestMoveToString:
    """测试 Move.to_string 方法。"""

    def test_standard_format(self) -> None:
        """测试标准格式输出。"""
        move: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        assert move.to_string() == "d10-d5-g7"

    def test_bottom_row(self) -> None:
        """测试第 1 行的输出。"""
        move: Move = Move(from_pos=(9, 0), to_pos=(0, 9), arrow_pos=(9, 0))
        assert move.to_string() == "a1-j10-a1"


class TestRoundtrip:
    """测试 from_string ↔ to_string 往返一致性。"""

    def test_roundtrip_various_moves(self) -> None:
        """测试多个走法的往返一致性。"""
        notations: list[str] = [
            "d10-d5-g7",
            "g10-g6-c10",
            "a7-a10-d7",
            "j7-d7-d10",
            "d1-d5-g8",
            "g1-d4-d8",
            "a1-j10-a1",
        ]
        for notation in notations:
            move: Move = Move.from_string(notation)
            assert move.to_string() == notation, f"往返失败: '{notation}' → '{move.to_string()}'"


class TestFromStringErrors:
    """测试非法输入的错误处理。"""

    def test_empty_string(self) -> None:
        """空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError):
            Move.from_string("")

    def test_missing_segment(self) -> None:
        """少一段应抛出 ValueError。"""
        with pytest.raises(ValueError):
            Move.from_string("a4-d4")

    def test_extra_segment(self) -> None:
        """多一段应抛出 ValueError。"""
        with pytest.raises(ValueError):
            Move.from_string("a4-d4-g7-x1")

    def test_invalid_coordinate(self) -> None:
        """非法坐标应抛出 ValueError。"""
        with pytest.raises(ValueError):
            Move.from_string("z9-d4-g7")


class TestMoveImmutability:
    """测试 Move 对象的不可变性。"""

    def test_frozen(self) -> None:
        """尝试修改属性应抛出异常。"""
        move: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        with pytest.raises(Exception):  # FrozenInstanceError
            move.from_pos = (1, 1)  # type: ignore[misc]


class TestMoveEquality:
    """测试 Move 对象的相等性。"""

    def test_equal(self) -> None:
        """相同内容的 Move 应相等。"""
        m1: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        m2: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        assert m1 == m2

    def test_not_equal(self) -> None:
        """不同内容的 Move 应不相等。"""
        m1: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        m2: Move = Move(from_pos=(0, 6), to_pos=(5, 6), arrow_pos=(3, 9))
        assert m1 != m2

    def test_hashable(self) -> None:
        """Move 应可哈希（可放入 set）。"""
        m1: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        m2: Move = Move(from_pos=(0, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        s: set[Move] = {m1, m2}
        assert len(s) == 1  # 去重
