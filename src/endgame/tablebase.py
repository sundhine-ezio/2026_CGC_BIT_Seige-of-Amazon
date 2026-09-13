"""残局库查询接口。

加载预计算的残局库（.pkl文件），按 Zobrist 哈希查询精确胜负结果。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

from src.game.constants import WIN as C_WIN
from src.game.constants import LOSE as C_LOSE


@dataclass
class EndgameResult:
    """残局查询结果。"""

    result: int       # WIN(1) / LOSE(-1)
    distance: int     # 距终局的步数


class Tablebase:
    """残局库——存储预计算的精确胜负结果。

    数据文件: pickle 格式，dict[int, dict[str, object]]
        键: Zobrist 哈希值
        值: {"result": "WIN"/"LOSE", "distance": int}
    """

    def __init__(self, data_path: str | None = None) -> None:
        """加载残局库。

        Args:
            data_path: .pkl 文件路径。若为 None 或文件不存在，残局库为空。

        Raises:
            FileNotFoundError: 指定了路径但文件不存在
        """
        self._data: dict[int, EndgameResult] = {}

        if data_path is None or not os.path.exists(data_path):
            self._loaded: bool = False
            return

        try:
            with open(data_path, "rb") as f:
                raw: dict[int, dict[str, object]] = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"残局库文件不存在: {data_path}")
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"残局库文件损坏: {e}") from e

        for hash_key, entry in raw.items():
            result_str: str = str(entry["result"])
            result_val: int = C_WIN if result_str == "WIN" else C_LOSE
            distance_val: int = int(entry.get("distance", 0))  # type: ignore[arg-type]
            self._data[hash_key] = EndgameResult(
                result=result_val, distance=distance_val
            )

        self._loaded: bool = True

    def query(self, board_hash: int) -> EndgameResult | None:
        """查询残局库。

        Args:
            board_hash: 64 位 Zobrist 哈希值

        Returns:
            EndgameResult 若命中，None 若未命中
        """
        if not self._loaded:
            return None
        return self._data.get(board_hash)

    def is_available(self) -> bool:
        """残局库是否已加载且非空。"""
        return self._loaded and len(self._data) > 0
