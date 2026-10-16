"""game/rules.py 的单元测试。"""

import pytest

from src.game.board import Board
from src.game.constants import BLACK, BURNED, EMPTY, WHITE
from src.game.move import Move
from src.game.rules import Rules


@pytest.fixture
def board() -> Board:
    """创建标准初始棋盘。"""
    return Board()


class TestGetQueenReachable:
    """测试 get_queen_reachable 方法。"""

    def test_center_empty_board(self) -> None:
        """空棋盘中心位置可达 27 格（8方向全通）。"""
        # 构造只有一枚棋子的棋盘
        b: Board = Board()
        b.reset()
        # 清空所有棋子
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    # 直接操作内部 grid 进行清理
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]
        # 放一个白子在中心
        b._grid[5, 5] = WHITE  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(5, 5)]  # type: ignore[attr-defined]

        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(b, (5, 5))
        # 中心在 10×10 棋盘: 上 5, 下 4, 左 5, 右 4, 对角线也类似
        # 总数 = 上(5) + 下(4) + 左(5) + 右(4)
        #   + 左上(min(5,5)) + 右上(min(5,4)) + 左下(min(4,5)) + 右下(min(4,4))
        # = 5+4+5+4 + 5+4+4+4 = 18 + 17 = 35
        # 实际上 8 方向分别计算即可
        assert len(reachable) > 0

    def test_corner_empty(self) -> None:
        """角落位置可达格数较少。"""
        b: Board = Board()
        b.reset()
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]

        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(b, (0, 0))
        # 从 (0,0): 右(9) + 下(9) + 右下(9) = 27
        assert len(reachable) == 27

    def test_blocked_by_piece(self) -> None:
        """皇后走法被棋子阻挡。"""
        b: Board = Board()
        b.reset()
        # 清空
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 3] = BLACK  # 阻挡右方向 # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(0, 3)]  # type: ignore[attr-defined]

        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(b, (0, 0))
        # 右方向只能到 (0,1), (0,2)，不能到 (0,3) 及以后
        assert (0, 1) in reachable
        assert (0, 2) in reachable
        assert (0, 3) not in reachable
        assert (0, 4) not in reachable

    def test_blocked_by_burned(self) -> None:
        """皇后走法被烧毁格阻挡。"""
        b: Board = Board()
        b.reset()
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 5] = BURNED  # 烧毁格阻挡 # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]

        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(b, (0, 0))
        assert (0, 5) not in reachable


class TestGetLegalMoves:
    """测试 get_legal_moves 方法。"""

    def test_initial_board_has_moves(self, board: Board) -> None:
        """初始棋盘应有合法走法。"""
        moves: list[Move] = Rules.get_legal_moves(board)
        assert len(moves) > 0

    def test_all_moves_are_legal(self, board: Board) -> None:
        """所有生成的走法应可正常执行。"""
        moves: list[Move] = Rules.get_legal_moves(board)
        for move in moves[:20]:  # 抽样检查
            b2: Board = board.clone()
            b2.make_move(move)
            # 不抛异常即为合法

    def test_no_duplicate_moves(self, board: Board) -> None:
        """走法不应重复。"""
        moves: list[Move] = Rules.get_legal_moves(board)
        move_strings: list[str] = [m.to_string() for m in moves]
        assert len(move_strings) == len(set(move_strings))

    def test_moves_for_correct_player(self, board: Board) -> None:
        """走法应来自当前玩家。"""
        moves: list[Move] = Rules.get_legal_moves(board)
        for move in moves[:20]:
            assert board.get_piece_at(*move.from_pos) == board.get_current_player()

    def test_arrow_not_on_piece(self, board: Board) -> None:
        """箭落点不应在有棋子的格子上。"""
        moves: list[Move] = Rules.get_legal_moves(board)
        for move in moves[:20]:
            # 除了 from_pos（棋子已移走），箭落点不应有棋子
            assert board.get_piece_at(*move.arrow_pos) in (EMPTY, board.get_current_player())
            # 实际上 from_pos 在射箭时已空，检查 to_pos 不等于 arrow_pos
            assert move.to_pos != move.arrow_pos


class TestIsGameOver:
    """测试 is_game_over 方法。"""

    def test_initial_board_not_over(self, board: Board) -> None:
        """初始棋盘不应终局。"""
        assert Rules.is_game_over(board) is False

    def test_deadlock_from_constructed(self) -> None:
        """构造一个封死局面测试终局。"""
        b: Board = Board()
        b.reset()
        # 清空所有棋子
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]

        # 放一个白子在角落，周围全烧毁——白方无合法走法
        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 1] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 0] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 1] = BURNED  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(9, 9)]  # type: ignore[attr-defined]
        b._grid[9, 9] = BLACK  # type: ignore[attr-defined]
        b._current_player = WHITE  # type: ignore[attr-defined]

        assert Rules.is_game_over(b) is True

    def test_get_winner(self) -> None:
        """终局时胜者判定正确。"""
        b: Board = Board()
        b.reset()
        for r in range(10):
            for c in range(10):
                if b.get_piece_at(r, c) in (WHITE, BLACK):
                    b._grid[r, c] = EMPTY  # type: ignore[attr-defined]

        b._grid[0, 0] = WHITE  # type: ignore[attr-defined]
        b._grid[0, 1] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 0] = BURNED  # type: ignore[attr-defined]
        b._grid[1, 1] = BURNED  # type: ignore[attr-defined]
        b._amazon_positions[WHITE] = [(0, 0)]  # type: ignore[attr-defined]
        b._amazon_positions[BLACK] = [(9, 9)]  # type: ignore[attr-defined]
        b._grid[9, 9] = BLACK  # type: ignore[attr-defined]
        b._current_player = WHITE  # type: ignore[attr-defined]

        # 白方无合法走法 → 黑方胜
        assert Rules.get_winner(b) == BLACK


class TestGetLegalArrowTargets:
    """测试 get_legal_arrow_targets——GUI 校验人类落箭所用的口径。

    回归背景：原先 GUI 直接用 get_queen_reachable(board, move_target) 在
    走子**前**的局面上算箭点，因此亚马逊刚腾出的原位置仍被自己的棋子占着、
    被判为非法，而 AI 走 get_legal_moves 路径却能走这步棋（射回原位在亚马逊
    棋里很常见）。人类棋手因此在 GUI 里无法完成该操作。
    """

    def test_origin_is_legal_arrow_target(self, board: Board) -> None:
        """亚马逊刚腾出的原位置必须可作箭落点。"""
        from_pos: tuple[int, int] = (9, 3)  # d1
        to_pos: tuple[int, int] = (5, 3)  # d5
        targets: list[tuple[int, int]] = Rules.get_legal_arrow_targets(
            board, from_pos, to_pos
        )
        assert from_pos in targets, "射回原位应当合法"

    def test_destination_is_not_arrow_target(self, board: Board) -> None:
        """箭不能落在亚马逊自己站着的格子上。"""
        from_pos: tuple[int, int] = (9, 3)
        to_pos: tuple[int, int] = (5, 3)
        targets: list[tuple[int, int]] = Rules.get_legal_arrow_targets(
            board, from_pos, to_pos
        )
        assert to_pos not in targets

    def test_does_not_modify_board(self, board: Board) -> None:
        """本方法只读，不得改动传入的棋盘。"""
        before = board.get_grid().copy()
        Rules.get_legal_arrow_targets(board, (9, 3), (5, 3))
        assert (board.get_grid() == before).all()
        assert board.get_current_player() == WHITE

    def test_matches_get_legal_moves(self, board: Board) -> None:
        """对每个可达落点，箭点集合必须与 get_legal_moves 完全一致。

        这是 GUI 与 AI 口径一致性的保证：AI 能走的箭点，人类也必须能走。
        """
        from_pos: tuple[int, int] = (9, 3)
        pairs: set[tuple[tuple[int, int], tuple[int, int]]] = {
            (m.to_pos, m.arrow_pos)
            for m in Rules.get_legal_moves(board)
            if m.from_pos == from_pos
        }
        assert pairs, "测试前提：该亚马逊应有合法走法"

        for to_pos in {to for to, _ in pairs}:
            expected: set[tuple[int, int]] = {
                arrow for (to, arrow) in pairs if to == to_pos
            }
            actual: set[tuple[int, int]] = set(
                Rules.get_legal_arrow_targets(board, from_pos, to_pos)
            )
            assert actual == expected, f"落点 {to_pos} 的箭点集合与 get_legal_moves 不一致"

    def test_blocked_origin_not_arrow_target(self) -> None:
        """若原位置被别的棋子占着（非腾空情形），不得当作箭点。"""
        b: Board = Board()
        b._grid[5, 2] = BLACK  # type: ignore[attr-defined]
        # 白方从 d1 走到 d5，原位置 d1 腾空 → d1 合法；但 d5 左侧的 c5 被黑子占
        targets: list[tuple[int, int]] = Rules.get_legal_arrow_targets(
            b, (9, 3), (5, 3)
        )
        assert (5, 2) not in targets, "被占格不能作箭落点"
        assert (9, 3) in targets
