"""game/constants.py 的单元测试。"""

import pytest
from src.game.constants import (
    BLACK,
    BOARD_SIZE,
    BURNED,
    EMPTY,
    INITIAL_BLACK_POSITIONS,
    INITIAL_WHITE_POSITIONS,
    QUEEN_DIRECTIONS,
    WHITE,
    algebraic_to_pos,
    is_in_bounds,
    pos_to_algebraic,
)


class TestPosToAlgebraic:
    """测试 pos_to_algebraic 函数。"""

    def test_corners(self) -> None:
        """测试四个角坐标的转换。"""
        assert pos_to_algebraic(0, 0) == "a10"
        assert pos_to_algebraic(0, 9) == "j10"
        assert pos_to_algebraic(9, 0) == "a1"
        assert pos_to_algebraic(9, 9) == "j1"

    def test_initial_positions(self) -> None:
        """测试所有初始棋子位置的转换。"""
        assert pos_to_algebraic(0, 3) == "d10"
        assert pos_to_algebraic(0, 6) == "g10"
        assert pos_to_algebraic(3, 0) == "a7"
        assert pos_to_algebraic(3, 9) == "j7"
        assert pos_to_algebraic(6, 0) == "a4"
        assert pos_to_algebraic(6, 9) == "j4"
        assert pos_to_algebraic(9, 3) == "d1"
        assert pos_to_algebraic(9, 6) == "g1"

    def test_center(self) -> None:
        """测试中心位置的转换。"""
        assert pos_to_algebraic(5, 4) == "e5"


class TestAlgebraicToPos:
    """测试 algebraic_to_pos 函数。"""

    def test_corners(self) -> None:
        """测试四个角坐标的解析。"""
        assert algebraic_to_pos("a10") == (0, 0)
        assert algebraic_to_pos("j10") == (0, 9)
        assert algebraic_to_pos("a1") == (9, 0)
        assert algebraic_to_pos("j1") == (9, 9)

    def test_single_digit_row(self) -> None:
        """测试单位数行号的解析。"""
        assert algebraic_to_pos("d5") == (5, 3)
        assert algebraic_to_pos("g7") == (3, 6)

    def test_initial_positions(self) -> None:
        """测试初始棋子位置的解析。"""
        assert algebraic_to_pos("d10") == (0, 3)
        assert algebraic_to_pos("g10") == (0, 6)
        assert algebraic_to_pos("a7") == (3, 0)
        assert algebraic_to_pos("j7") == (3, 9)
        assert algebraic_to_pos("a4") == (6, 0)
        assert algebraic_to_pos("j4") == (6, 9)
        assert algebraic_to_pos("d1") == (9, 3)
        assert algebraic_to_pos("g1") == (9, 6)


class TestCoordinateRoundTrip:
    """测试坐标转换的往返一致性。"""

    def test_all_initial_positions(self) -> None:
        """所有初始棋子位置往返一致。"""
        all_positions: list[tuple[int, int]] = (
            INITIAL_BLACK_POSITIONS + INITIAL_WHITE_POSITIONS
        )
        for row, col in all_positions:
            notation: str = pos_to_algebraic(row, col)
            result: tuple[int, int] = algebraic_to_pos(notation)
            assert result == (row, col), f"往返失败: ({row},{col}) → '{notation}' → {result}"

    def test_all_board_cells(self) -> None:
        """遍历全部 100 格验证往返一致性。"""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                notation = pos_to_algebraic(row, col)
                result = algebraic_to_pos(notation)
                assert result == (row, col)


class TestAlgebraicToPosErrors:
    """测试非法输入的错误处理。"""

    def test_empty_string(self) -> None:
        """空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("")

    def test_too_short(self) -> None:
        """过短字符串应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("a")

    def test_invalid_column(self) -> None:
        """非法列号应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("z1")

    def test_invalid_row(self) -> None:
        """非法行号应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("ax")

    def test_row_out_of_range_zero(self) -> None:
        """行号 0 越界应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("a0")

    def test_row_out_of_range_eleven(self) -> None:
        """行号 11 越界应抛出 ValueError。"""
        with pytest.raises(ValueError):
            algebraic_to_pos("a11")


class TestIsInBounds:
    """测试 is_in_bounds 函数。"""

    def test_valid_positions(self) -> None:
        """测试合法位置。"""
        assert is_in_bounds(0, 0) is True
        assert is_in_bounds(9, 9) is True
        assert is_in_bounds(5, 5) is True

    def test_negative_row(self) -> None:
        """负行号应返回 False。"""
        assert is_in_bounds(-1, 0) is False

    def test_negative_col(self) -> None:
        """负列号应返回 False。"""
        assert is_in_bounds(0, -1) is False

    def test_row_too_large(self) -> None:
        """超出行号应返回 False。"""
        assert is_in_bounds(10, 0) is False

    def test_col_too_large(self) -> None:
        """超出列号应返回 False。"""
        assert is_in_bounds(0, 10) is False


class TestConstants:
    """测试常量值。"""

    def test_board_size(self) -> None:
        """棋盘尺寸应为 10。"""
        assert BOARD_SIZE == 10

    def test_cell_values_distinct(self) -> None:
        """格位编码应互不相同。"""
        values: list[int] = [EMPTY, WHITE, BLACK, BURNED]
        assert len(values) == len(set(values))

    def test_queen_directions_count(self) -> None:
        """皇后走法应有 8 个方向。"""
        assert len(QUEEN_DIRECTIONS) == 8

    def test_queen_directions_unique(self) -> None:
        """皇后走法方向应无重复。"""
        assert len(set(QUEEN_DIRECTIONS)) == 8

    def test_initial_black_positions_count(self) -> None:
        """黑方应有 4 枚棋子。"""
        assert len(INITIAL_BLACK_POSITIONS) == 4

    def test_initial_white_positions_count(self) -> None:
        """白方应有 4 枚棋子。"""
        assert len(INITIAL_WHITE_POSITIONS) == 4

    def test_black_positions_in_bounds(self) -> None:
        """所有黑方初始位置应在棋盘范围内。"""
        for row, col in INITIAL_BLACK_POSITIONS:
            assert is_in_bounds(row, col)

    def test_white_positions_in_bounds(self) -> None:
        """所有白方初始位置应在棋盘范围内。"""
        for row, col in INITIAL_WHITE_POSITIONS:
            assert is_in_bounds(row, col)

    def test_no_overlap(self) -> None:
        """黑白初始位置不应重叠。"""
        black_set: set[tuple[int, int]] = set(INITIAL_BLACK_POSITIONS)
        white_set: set[tuple[int, int]] = set(INITIAL_WHITE_POSITIONS)
        assert black_set.isdisjoint(white_set)
