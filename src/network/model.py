"""策略-价值双头走法打分网络（MCTS+ 第二阶段 v3）。

实现 AlphaZero 风格残差塔 + 走法打分（CandidateMoveNet）：
  - 棋盘编码器: stem + 残差塔 → 全局池化 → board_feature 向量
  - 走法编码器: 每个候选走法 (from,to,arrow) → 6 维特征 → move_embedding
  - 打分头:     concat(board_feature, move_embedding) → 每候选一个标量 logit
  - 价值头:     board_feature → Tanh ∈ [-1, 1]

策略 = softmax(打分头 logits)，在 ai/mcts.py 中完成；本模块只负责网络前向与
单局面/批量推理接口。走法特征编码见 network/move_encoding.py。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # torch 惯用别名

from src.game.board import Board
from src.game.constants import BURNED, EMPTY, WHITE
from src.game.move import Move
from src.network.config import NetworkConfig
from src.network.move_encoding import MOVE_FEATURE_DIM, encode_legal_moves

#: 非法（补零）走法的屏蔽 logit，FP16 安全，softmax 后近似 0。
MASKED_LOGIT: float = -3e4


class ResidualBlock(nn.Module):
    """残差块：两个 3×3 卷积 + BN + 跳跃连接。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self._conv1: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self._bn1: nn.BatchNorm2d = nn.BatchNorm2d(channels)
        self._conv2: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self._bn2: nn.BatchNorm2d = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差前向：ReLU(x + BN(Conv(ReLU(BN(Conv(x))))))。"""
        out: torch.Tensor = F.relu(self._bn1(self._conv1(x)))
        out = self._bn2(self._conv2(out))
        return F.relu(x + out)


class AmazonsNet(nn.Module):
    """策略-价值双头走法打分网络。"""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        """按 NetworkConfig 构建网络层。

        Args:
            config: 网络超参数配置；None 时使用默认值。
        """
        super().__init__()
        self._config: NetworkConfig = config or NetworkConfig()
        cfg: NetworkConfig = self._config

        # 初始卷积: (B, C, 10, 10) -> (B, F, 10, 10)
        self._stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(cfg.input_channels, cfg.num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.num_filters),
            nn.ReLU(inplace=True),
        )

        # 残差塔: num_residual_blocks × ResidualBlock
        self._tower: nn.Sequential = nn.Sequential(
            *[ResidualBlock(cfg.num_filters) for _ in range(cfg.num_residual_blocks)]
        )

        # 棋盘特征: 全局池化 → Flatten → FC → board_feature (B, D)
        self._board_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self._board_fc: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.num_filters, cfg.board_feature_dim),
            nn.ReLU(inplace=True),
        )

        # 走法编码器: 6 维特征 → move_embedding (B, M, E)
        self._move_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(MOVE_FEATURE_DIM, cfg.move_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.move_embedding_dim, cfg.move_embedding_dim),
            nn.ReLU(inplace=True),
        )

        # 打分头: concat(board_feature, move_embedding) → 每候选一个标量
        self._score_head: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.board_feature_dim + cfg.move_embedding_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # 价值头: board_feature → Tanh ∈ [-1, 1]
        self._value_head: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.board_feature_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Tanh(),
        )

    @property
    def config(self) -> NetworkConfig:
        """返回网络架构配置（供并行 self-play 构建 worker 模型复用）。"""
        return self._config

    def forward(
        self,
        observation: torch.Tensor,
        move_features: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            observation:  (B, input_channels, 10, 10) 棋盘状态张量
            move_features: (B, M, MOVE_FEATURE_DIM) 候选走法特征
            move_mask:    (B, M) 合法走法掩码（1 合法 / 0 补零），可为 None

        Returns:
            {
                "policy_logits": (B, M) 每个候选走法的打分 logits
                "value":         (B, 1) 胜率 ∈ [-1, 1]（当前行棋方视角）
            }
        """
        h: torch.Tensor = self._tower(self._stem(observation))
        board_feature: torch.Tensor = self._board_fc(
            self._board_pool(h).flatten(1)
        )  # (B, D)
        move_embedding: torch.Tensor = self._move_encoder(move_features)  # (B, M, E)

        move_count: int = move_features.shape[1]
        expanded: torch.Tensor = board_feature.unsqueeze(1).expand(
            -1, move_count, -1
        )  # (B, M, D)
        score_input: torch.Tensor = torch.cat([expanded, move_embedding], dim=-1)
        policy_logits: torch.Tensor = self._score_head(score_input).squeeze(-1)  # (B, M)
        if move_mask is not None:
            policy_logits = policy_logits.masked_fill(move_mask <= 0, MASKED_LOGIT)

        value: torch.Tensor = self._value_head(board_feature)  # (B, 1)
        return {"policy_logits": policy_logits, "value": value}

    @staticmethod
    def board_to_tensor(board: Board) -> np.ndarray:
        """将 Board 转换为 (input_channels, 10, 10) numpy 数组（当前行棋方视角）。

        通道（均为 0/1 二值）:
          [0] 己方亚马逊: grid == current_player
          [1] 对方亚马逊: grid == -current_player
          [2] 烧毁格:     grid == BURNED
          [3] 空格:       grid == EMPTY
          [4] 行棋方指示: 白方行棋 → 全 1，黑方行棋 → 全 0
          [5] 最近一步标记: 保留通道（暂未接入，恒为 0）

        Args:
            board: 棋盘状态

        Returns:
            (6, 10, 10) float32 数组
        """
        grid: np.ndarray = board.get_grid()
        player: int = board.get_current_player()

        channels: list[np.ndarray] = [
            (grid == player).astype(np.float32),
            (grid == -player).astype(np.float32),
            (grid == BURNED).astype(np.float32),
            (grid == EMPTY).astype(np.float32),
            np.full(grid.shape, 1.0 if player == WHITE else 0.0, dtype=np.float32),
            np.zeros(grid.shape, dtype=np.float32),
        ]
        return np.stack(channels, axis=0)

    def predict(self, board: Board, moves: Sequence[Move]) -> tuple[np.ndarray, float]:
        """单局面推理（供 MCTS 调用），对给定合法走法打分。

        Args:
            board: 棋盘状态
            moves: 该局面下的合法走法列表（顺序与返回 logits 对齐）

        Returns:
            (policy_logits, value)
            policy_logits: (len(moves),) 每个走法的打分 logits
            value:         当前行棋方视角，∈ [-1, 1]
        """
        device: torch.device = next(self.parameters()).device
        board_tensor: torch.Tensor = (
            torch.from_numpy(self.board_to_tensor(board)).unsqueeze(0).to(device)
        )
        move_tensor: torch.Tensor = (
            torch.from_numpy(encode_legal_moves(moves)).unsqueeze(0).to(device)
        )

        was_training: bool = self.training
        self.eval()
        try:
            with torch.no_grad():
                out: dict[str, torch.Tensor] = self.forward(board_tensor, move_tensor)
        finally:
            self.train(was_training)

        logits: np.ndarray = out["policy_logits"].squeeze(0).cpu().numpy()
        value: float = float(out["value"].squeeze().cpu().numpy())
        return logits, value

    def predict_batch(
        self, boards: list[Board], moves_list: Sequence[Sequence[Move]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """批量局面推理（供 GPU 批量 MCTS 使用），对每局合法走法打分。

        各局合法走法数可能不同；内部按 max(M) 补零并用掩码屏蔽。

        Args:
            boards:     一批棋盘（各自视角独立）
            moves_list: 与 boards 对齐的合法走法列表

        Returns:
            (logits_list, values)
            logits_list: 与 boards 对齐，每项为 (M_i,) 的打分 logits
            values:      (B,) float32，各自当前行棋方视角的价值 ∈ [-1, 1]
        """
        if not boards:
            return [], np.zeros((0,), dtype=np.float32)
        device: torch.device = next(self.parameters()).device
        board_batch: torch.Tensor = torch.from_numpy(
            np.stack([self.board_to_tensor(b) for b in boards])
        ).to(device)

        max_m: int = max(len(ms) for ms in moves_list)
        move_features = np.zeros(
            (len(boards), max_m, MOVE_FEATURE_DIM), dtype=np.float32
        )
        move_mask = np.zeros((len(boards), max_m), dtype=np.float32)
        for i, moves in enumerate(moves_list):
            m_i: int = len(moves)
            move_features[i, :m_i] = encode_legal_moves(moves)
            move_mask[i, :m_i] = 1.0

        move_tensor: torch.Tensor = torch.from_numpy(move_features).to(device)
        mask_tensor: torch.Tensor = torch.from_numpy(move_mask).to(device)

        was_training: bool = self.training
        self.eval()
        try:
            with torch.no_grad():
                out: dict[str, torch.Tensor] = self.forward(
                    board_batch, move_tensor, mask_tensor
                )
        finally:
            self.train(was_training)

        policy: np.ndarray = out["policy_logits"].cpu().numpy()  # (B, max_m)
        values: np.ndarray = out["value"].squeeze(-1).cpu().numpy()  # (B,)
        logits_list: list[np.ndarray] = [
            policy[i, : len(moves)] for i, moves in enumerate(moves_list)
        ]
        return logits_list, values
