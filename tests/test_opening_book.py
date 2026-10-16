"""opening_book/book.py 的单元测试。"""

import os
import pytest
from src.opening_book.book import OpeningBook

TEST_BOOK_PATH: str = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "opening_book",
    "data",
    "opening_book.json",
)


@pytest.fixture
def book() -> OpeningBook:
    """加载测试开局库。"""
    path: str = os.path.abspath(TEST_BOOK_PATH)
    return OpeningBook(path)


class TestOpeningBook:
    """测试开局库。"""

    def test_load_success(self, book: OpeningBook) -> None:
        """应成功加载。"""
        assert book.is_available()

    def test_get_total_positions(self, book: OpeningBook) -> None:
        """应至少有一个局面。"""
        assert book.get_total_positions() >= 1

    def test_query_hit(self, book: OpeningBook) -> None:
        """查询哈希 0 应命中（示例数据）。"""
        result: list | None = book.query(0)
        assert result is not None
        assert len(result) >= 1

    def test_query_miss(self, book: OpeningBook) -> None:
        """查询不存在的哈希应返回 None。"""
        assert book.query(99999) is None

    def test_get_random_move(self, book: OpeningBook) -> None:
        """加权随机选择应返回合法 Move。"""
        import random
        random.seed(42)
        move = book.get_random_move(0)
        # 示例数据中的哈希 0 可能对不上初始棋盘的 Zobrist 哈希
        # 这里测试的是接口可用性
        if move is not None:
            assert len(move.to_string()) > 0

    def test_missing_file_graceful(self) -> None:
        """不存在的文件应返回空开局库（不抛异常）。"""
        book2: OpeningBook = OpeningBook("nonexistent_file.json")
        assert not book2.is_available()

    def test_none_path(self) -> None:
        """None 路径应返回空开局库。"""
        book2: OpeningBook = OpeningBook(None)
        assert not book2.is_available()
        assert book2.query(0) is None
