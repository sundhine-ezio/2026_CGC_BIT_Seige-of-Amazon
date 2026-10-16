"""main.py 比赛模式时间分配测试。"""

from __future__ import annotations

import pytest

from main import (
    COMPETITION_EXPECTED_MOVES,
    COMPETITION_SAFETY,
    _competition_move_budget_ms,
)

TOTAL_MS = 900_000.0  # 15 分钟


def test_opening_budget() -> None:
    """开局首步 = 总时间 / 预估步数 × 安全系数。"""
    budget = _competition_move_budget_ms(0.0, 0, TOTAL_MS)
    assert budget == pytest.approx(TOTAL_MS * COMPETITION_SAFETY / COMPETITION_EXPECTED_MOVES)


def test_never_exceeds_total_budget() -> None:
    """整局模拟：每步都花光分到的预算，累计用时不得超过总预算。

    这是本方案的核心保证——旧实现固定每步 total/100 且不跟踪已用时间，
    一局超过 125 步就会突破 15 分钟总时限。
    """
    elapsed = 0.0
    moves = 0
    while elapsed < TOTAL_MS and moves < 2000:
        budget = _competition_move_budget_ms(elapsed, moves, TOTAL_MS)
        assert budget <= max(TOTAL_MS - elapsed, 0.0) + 1e-9, "单步预算超过剩余时间"
        elapsed += budget
        moves += 1
    assert elapsed <= TOTAL_MS, f"累计 {elapsed}ms 超过总预算 {TOTAL_MS}ms"
    # 确认这个模拟确实下到了"异常长"的局面
    assert moves > 125, f"只模拟到 {moves} 步，不足以覆盖旧实现的超时场景"


def test_budget_stays_in_sane_band() -> None:
    """全局面单步预算都应是可用值：既不为 0，也不会一步吃掉整局时间。"""
    for moves in (0, 50, 100, 200, 500):
        for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
            budget = _competition_move_budget_ms(TOTAL_MS * frac, moves, TOTAL_MS)
            assert 0 < budget <= TOTAL_MS * 0.2, (moves, frac, budget)


def test_more_elapsed_time_means_less_budget() -> None:
    """同一局面下，已用时间越多，单步预算越小。"""
    early = _competition_move_budget_ms(0.0, 0, TOTAL_MS)
    late = _competition_move_budget_ms(TOTAL_MS * 0.9, 0, TOTAL_MS)
    assert late < early


def test_always_positive() -> None:
    """绝不能返回 0 或负数：AlphaBetaEngine 把 <=0 当成"不限时"。"""
    assert _competition_move_budget_ms(0.0, 0, TOTAL_MS) > 0
    assert _competition_move_budget_ms(TOTAL_MS, 0, TOTAL_MS) > 0
    assert _competition_move_budget_ms(TOTAL_MS * 10, 999, TOTAL_MS) > 0
