"""ai/engine.py 的单元测试。"""

import pytest
from src.ai.engine import Engine
from src.game.board import Board
from src.game.move import Move


class TestEngineABC:
    """测试 Engine 抽象基类。"""

    def test_cannot_instantiate(self) -> None:
        """实例化抽象基类应抛出 TypeError。"""
        with pytest.raises(TypeError):
            Engine()  # type: ignore[abstract]

    def test_missing_search(self) -> None:
        """未实现 search 方法的子类不能实例化。"""

        class BadEngine(Engine):
            def get_name(self) -> str:
                return "bad"

            def reset(self) -> None:
                pass

        with pytest.raises(TypeError):
            BadEngine()  # type: ignore[abstract]

    def test_can_instantiate_full_impl(self) -> None:
        """全部方法实现的子类可实例化。"""

        class GoodEngine(Engine):
            def search(self, board: Board, time_budget_ms: float) -> Move:  # type: ignore[override]
                legal = __import__("src.game.rules", fromlist=["Rules"]).Rules.get_legal_moves(board)
                return legal[0]

            def get_name(self) -> str:
                return "good"

            def reset(self) -> None:
                pass

        engine: GoodEngine = GoodEngine()
        assert engine.get_name() == "good"
