"""game/board.py 的单元测试。"""

import numpy as np
import pytest
from src.game.board import Board
from src.game.constants import (
    BLACK,
    BURNED,
    EMPTY,
    INITIAL_BLACK_POSITIONS,
    INITIAL_WHITE_POSITIONS,
    WHITE,
)
from src.game.move import Move


@pytest.fixture
def board() -> Board:
    """创建标准初始棋盘。"""
    return Board()


@pytest.fixture
def sample_move() -> Move:
    """白方 d1→d5→g7 走法。"""
    return Move(from_pos=(9, 3), to_pos=(5, 3), arrow_pos=(3, 6))


class TestBoardInit:
    """测试棋盘初始化。"""

    def test_grid_shape(self, board: Board) -> None:
        """棋盘应为 10×10。"""
        grid: np.ndarray = board.get_grid()
        assert grid.shape == (10, 10)

    def test_initial_black_positions(self, board: Board) -> None:
        """黑方棋子应在正确位置。"""
        for r, c in INITIAL_BLACK_POSITIONS:
            assert board.get_piece_at(r, c) == BLACK

    def test_initial_white_positions(self, board: Board) -> None:
        """白方棋子应在正确位置。"""
        for r, c in INITIAL_WHITE_POSITIONS:
            assert board.get_piece_at(r, c) == WHITE

    def test_empty_squares(self, board: Board) -> None:
        """初始棋盘应有 92 个空格（100 - 8 棋子）。"""
        grid: np.ndarray = board.get_grid()
        assert np.count_nonzero(grid == EMPTY) == 92

    def test_current_player_is_white(self, board: Board) -> None:
        """白方先行。"""
        assert board.get_current_player() == WHITE

    def test_amazon_count(self, board: Board) -> None:
        """双方各有 4 枚亚马逊。"""
        assert len(board.get_amazon_positions(WHITE)) == 4
        assert len(board.get_amazon_positions(BLACK)) == 4


class TestMakeMove:
    """测试 make_move 方法。"""

    def test_piece_moved(self, board: Board, sample_move: Move) -> None:
        """棋子从 from_pos 移至 to_pos。"""
        board.make_move(sample_move)
        assert board.get_piece_at(9, 3) == EMPTY   # d1 空了
        assert board.get_piece_at(5, 3) == WHITE    # d5 有白子

    def test_arrow_burned(self, board: Board, sample_move: Move) -> None:
        """箭落点变为烧毁格。"""
        board.make_move(sample_move)
        assert board.get_piece_at(3, 6) == BURNED   # g7 烧毁

    def test_player_switched(self, board: Board, sample_move: Move) -> None:
        """走子后切换玩家。"""
        board.make_move(sample_move)
        assert board.get_current_player() == BLACK

    def test_amazon_positions_updated(self, board: Board, sample_move: Move) -> None:
        """棋子位置表更新。"""
        board.make_move(sample_move)
        white_positions: list[tuple[int, int]] = board.get_amazon_positions(WHITE)
        assert (9, 3) not in white_positions
        assert (5, 3) in white_positions
        assert len(white_positions) == 4

    def test_grid_copy_independent(self, board: Board, sample_move: Move) -> None:
        """get_grid 返回拷贝，不随 make_move 改变。"""
        grid_before: np.ndarray = board.get_grid()
        board.make_move(sample_move)
        grid_after: np.ndarray = board.get_grid()
        # 拷贝应不同
        assert not np.array_equal(grid_before, grid_after)
        # 原始拷贝不受影响
        assert grid_before[9, 3] == WHITE


class TestUndoMove:
    """测试 undo_move 方法。"""

    def test_undo_restores_piece(self, board: Board, sample_move: Move) -> None:
        """撤销后棋子回到原位。"""
        board.make_move(sample_move)
        board.undo_move()
        assert board.get_piece_at(9, 3) == WHITE   # d1 恢复
        assert board.get_piece_at(5, 3) == EMPTY    # d5 变空

    def test_undo_restores_burned(self, board: Board, sample_move: Move) -> None:
        """撤销后烧毁格恢复。"""
        board.make_move(sample_move)
        board.undo_move()
        assert board.get_piece_at(3, 6) == EMPTY    # g7 恢复

    def test_undo_restores_player(self, board: Board, sample_move: Move) -> None:
        """撤销后恢复走子方。"""
        player_before: int = board.get_current_player()
        board.make_move(sample_move)
        board.undo_move()
        assert board.get_current_player() == player_before

    def test_undo_restores_hash(self, board: Board, sample_move: Move) -> None:
        """撤销后哈希值恢复。"""
        hash_before: int = board.get_zobrist_hash()
        board.make_move(sample_move)
        board.undo_move()
        assert board.get_zobrist_hash() == hash_before

    def test_undo_restores_amazon_positions(self, board: Board, sample_move: Move) -> None:
        """撤销后棋子位置表恢复。"""
        positions_before: list[tuple[int, int]] = sorted(
            board.get_amazon_positions(WHITE)
        )
        board.make_move(sample_move)
        board.undo_move()
        positions_after: list[tuple[int, int]] = sorted(
            board.get_amazon_positions(WHITE)
        )
        assert positions_after == positions_before

    def test_undo_empty_history_raises(self, board: Board) -> None:
        """空历史时撤销应抛 IndexError。"""
        with pytest.raises(IndexError):
            board.undo_move()

    def test_undo_ten_random_moves(self, board: Board) -> None:
        """连续 10 次随机走法 + 撤销应完全恢复。"""
        # 使用简单走法序列
        moves_to_test: list[Move] = []
        board_test: Board = Board()

        from src.game.rules import Rules

        for _ in range(10):
            legal_moves: list[Move] = Rules.get_legal_moves(board_test)
            if not legal_moves:
                break
            move: Move = legal_moves[0]
            moves_to_test.append(move)
            board_test.make_move(move)

        # 记录当前状态
        grid_before_undo: np.ndarray = board_test.get_grid()
        hash_before_undo: int = board_test.get_zobrist_hash()
        player_before_undo: int = board_test.get_current_player()

        # 全部撤销
        for _ in range(len(moves_to_test)):
            board_test.undo_move()

        # 应回到初始状态
        initial: Board = Board()
        assert np.array_equal(board_test.get_grid(), initial.get_grid())
        assert board_test.get_current_player() == WHITE
        assert sorted(board_test.get_amazon_positions(WHITE)) == sorted(
            initial.get_amazon_positions(WHITE)
        )


class TestClone:
    """测试 clone 方法。"""

    def test_clone_independent(self, board: Board) -> None:
        """修改克隆不影响原棋盘。"""
        clone: Board = board.clone()
        test_move: Move = Move(from_pos=(9, 3), to_pos=(5, 3), arrow_pos=(3, 6))
        clone.make_move(test_move)

        # 原棋盘不变
        assert board.get_piece_at(9, 3) == WHITE
        assert board.get_current_player() == WHITE

    def test_clone_identical_state(self, board: Board) -> None:
        """克隆后状态一致。"""
        clone: Board = board.clone()
        assert np.array_equal(clone.get_grid(), board.get_grid())
        assert clone.get_current_player() == board.get_current_player()
        assert clone.get_zobrist_hash() == board.get_zobrist_hash()


class TestReset:
    """测试 reset 方法。"""

    def test_reset_to_initial(self, board: Board) -> None:
        """走几步后 reset 应回到初始状态。"""
        board.make_move(Move(from_pos=(9, 3), to_pos=(5, 3), arrow_pos=(3, 6)))
        board.make_move(Move(from_pos=(0, 3), to_pos=(3, 5), arrow_pos=(5, 7)))
        board.reset()

        initial: Board = Board()
        assert np.array_equal(board.get_grid(), initial.get_grid())
        assert board.get_current_player() == WHITE
        assert len(board.get_amazon_positions(WHITE)) == 4

    def test_reset_clears_history(self, board: Board) -> None:
        """reset 后应清空走法历史。"""
        board.make_move(Move(from_pos=(9, 3), to_pos=(5, 3), arrow_pos=(3, 6)))
        board.reset()
        with pytest.raises(IndexError):
            board.undo_move()
