# game-rules 任务清单

> **模块**：`src/game/rules.py`  
> **依赖**：`game-constants`, `game-board`, `game-move`  
> **预估工时**：2 小时  

## 实现任务

- [ ] **R-01** 创建 `src/game/rules.py`，导入 `Board`, `Move`, `constants`

- [ ] **R-02** 实现 `Rules.get_queen_reachable(board, from_pos) → list[tuple]`
  - 对 `QUEEN_DIRECTIONS` 的 8 个方向逐一滑动
  - 每方向从 `from_pos` 出发，步进直到：
    - 越界（`not is_in_bounds`）→ 停止
    - 遇到非空格（`≠ EMPTY`）→ 停止
  - 只收集值为 `EMPTY` 的格子
  - 返回可达格子的 `(row, col)` 列表

- [ ] **R-03** 实现 `Rules.get_legal_moves(board) → list[Move]`
  - 获取当前玩家的 4 个亚马逊位置
  - 对每个亚马逊 A：
    1. 调用 `get_queen_reachable(board, A)` 获取移动目标候选
    2. 对每个移动目标 T：
       - **临时模拟移动**：将 A 处的棋子移到 T（设 A 为 EMPTY，T 为棋子）
       - 调用 `get_queen_reachable(board, T)` 获取箭落点候选
       - **恢复**：将棋子移回 A
       - 对每个箭落点 R，生成 `Move(A, T, R)`
  - 返回所有合法走法列表

  > ⚠️ 临时模拟移动可用 `make_move` + `undo_move`（稳妥但较慢），后续优化为 `make_partial` + `undo_partial`

- [ ] **R-04** 实现 `Rules.is_game_over(board) → bool`
  - 调用 `get_legal_moves(board)`，若返回空列表则为 `True`

- [ ] **R-05** 实现 `Rules.get_winner(board) → int`
  - 前提：`is_game_over(board)` 为 True
  - 返回对方玩家：`-board.get_current_player()`
  - 即 `current_player` 无合法走法，对方胜

## 测试任务

- [ ] **T-01** 创建 `tests/test_rules.py`
- [ ] **T-02** 测试 `get_queen_reachable`——空棋盘中心
  - 手工构造只有一枚棋子的棋盘
  - 从 (5,5) 出发应可达 27 格（8方向全通）
  - 从 (0,0) 出发应可达 21 格（3方向）
- [ ] **T-03** 测试 `get_queen_reachable`——有障碍物
  - 在路径上放置烧毁格，验证搜索正确停止
- [ ] **T-04** 测试 `get_legal_moves`——初始棋盘
  - 白方初始走法数量应在合理范围（可手工估算或参考已知值）
  - 每个返回的 Move 都能被 `board.make_move()` 正常执行
- [ ] **T-05** 测试 `get_legal_moves`——执行一步后的棋盘
  - 白方走完第一步后，黑方的合法走法数量和正确性
- [ ] **T-06** 测试 `is_game_over`——中间局面返回 False
- [ ] **T-07** 测试 `is_game_over`——构造死局返回 True
  - 将一方亚马逊完全围住（周围全是棋子或烧毁格）
- [ ] **T-08** 测试 `get_winner`——死局返回对方胜
- [ ] **T-09** 测试走法生成正确性：不包含移动终点有棋子的走法、不包含穿过障碍的走法
- [ ] **T-10** 运行 `python -m pytest tests/test_rules.py -v`，全部通过

## 完成标准
- [ ] 初始棋盘走法生成正常（数量合理、走法合法）
- [ ] 皇后走法在空旷/有障碍情况下均正确
- [ ] 终局判断准确
- [ ] 全部测试通过
