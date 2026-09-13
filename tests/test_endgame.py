"""endgame/ 模块的单元测试。"""

import os
import pickle
import pytest
from src.endgame.retrograde import RetrogradeGenerator
from src.endgame.tablebase import EndgameResult, Tablebase


class TestEndgameResult:
    """测试 EndgameResult 数据类。"""

    def test_create(self) -> None:
        """测试创建 EndgameResult。"""
        result: EndgameResult = EndgameResult(result=1, distance=3)
        assert result.result == 1
        assert result.distance == 3


class TestTablebase:
    """测试残局库查询。"""

    def test_none_path(self) -> None:
        """None 路径应返回空残局库。"""
        tb: Tablebase = Tablebase(None)
        assert not tb.is_available()
        assert tb.query(0) is None

    def test_missing_file_graceful(self) -> None:
        """不存在的文件应返回空残局库（不抛异常）。"""
        tb: Tablebase = Tablebase("nonexistent.pkl")
        assert not tb.is_available()

    def test_load_empty(self, tmp_path: str) -> None:
        """加载空残局库。"""
        import pickle
        # Test with a valid pickled dict
        pkl_path: str = os.path.join(str(tmp_path), "test.pkl")

        # Create a proper endgame database
        data: dict[int, dict[str, object]] = {
            42: {"result": "WIN", "distance": 3},
            99: {"result": "LOSE", "distance": 1},
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        tb: Tablebase = Tablebase(pkl_path)
        assert tb.is_available()

        result: EndgameResult | None = tb.query(42)
        assert result is not None
        assert result.result == 1  # WIN
        assert result.distance == 3

        result2: EndgameResult | None = tb.query(99)
        assert result2 is not None
        assert result2.result == -1  # LOSE

    def test_query_miss(self, tmp_path: str) -> None:
        """未命中应返回 None。"""
        pkl_path: str = os.path.join(str(tmp_path), "test2.pkl")
        data: dict[int, dict[str, object]] = {42: {"result": "WIN", "distance": 1}}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        tb: Tablebase = Tablebase(pkl_path)
        assert tb.query(999) is None


class TestRetrogradeGenerator:
    """测试回溯分析生成器。"""

    def test_estimate(self) -> None:
        """测试状态空间估算。"""
        gen: RetrogradeGenerator = RetrogradeGenerator(max_pieces=3, max_empty=15)
        count: int = gen.estimate_state_space()
        assert count > 0

    def test_generate_placeholder(self, tmp_path: str) -> None:
        """测试占位生成。"""
        gen: RetrogradeGenerator = RetrogradeGenerator(max_pieces=2, max_empty=4)
        output_path: str = os.path.join(str(tmp_path), "tablebase.pkl")
        result: dict = gen.generate(output_path)

        assert os.path.exists(output_path)
        assert len(result) > 0
