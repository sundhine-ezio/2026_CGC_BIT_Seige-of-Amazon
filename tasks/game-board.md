# game-board 任务清单

> **模块**：`src/game/board.py`  
> **依赖**：`game-constants`, `game-move`  
> **预估工时**：1.5 小时  

## 实现任务

- [ ] **B-01** 创建 `src/game/board.py`，导入 `numpy`, `Move`, `constants`
- [ ] **B-02** 定义 `UndoInfo` 内部数据类（用于 `move_history` 栈）
  ```python
  from_pos, to_pos, arrow_pos: tuple
  captured_from, captured_to, captured_arrow: int  # 被覆盖格子的原值
  prev_hash: int        # 走法前的 Zobrist 哈希
  prev_player: int      # 走法前的 current_player
  ```
- [ ] **B-03** 实现 `Board.__init__()`
  - 创建 `(10,10)` 全 `EMPTY` 的 numpy 数组（`dtype=np.int8`）
  - 按 `INITIAL_BLACK_POSITIONS` 放置 `BLACK`，按 `INITIAL_WHITE_POSITIONS` 放置 `WHITE`
  - `current_player = WHITE`（白方先行）
  - 初始化 `amazon_positions = {WHITE: [...], BLACK: [...]}`
  - `move_history = []`
  - 计算初始 Zobrist 哈希（暂时设为 0，等 `ai-transposition` 模块完成后再接入）

- [ ] **B-04** 实现 `get_grid() → np.ndarray`：返回 grid（注意：返回拷贝以避免外部修改）
- [ ] **B-05** 实现 `get_piece_at(row, col) → int`：返回 `grid[row][col]`
- [ ] **B-06** 实现 `get_current_player() → int`
- [ ] **B-07** 实现 `get_amazon_positions(player) → list[tuple]`：返回副本
- [ ] **B-08** 实现 `get_zobrist_hash() → int`

- [ ] **B-09** 实现 `make_move(move: Move) → None`
  1. 创建 `UndoInfo` 记录当前状态（3个被覆盖格的原值、当前哈希、当前玩家）
  2. `grid[from_pos] = EMPTY`
  3. `grid[to_pos] = piece`（原 from_pos 处的棋子值）
  4. `grid[arrow_pos] = BURNED`
  5. 更新 `amazon_positions`
  6. 更新 `zobrist_hash`（如已集成 Zobrist）
  7. `current_player = -current_player`
  8. `move_history.append(undo_info)`

- [ ] **B-10** 实现 `undo_move() → None`
  1. 从 `move_history` 弹出最后一条 `UndoInfo`
  2. 恢复 `grid` 三个格子的原值
  3. 恢复 `amazon_positions`
  4. 恢复 `zobrist_hash` 和 `current_player`

- [ ] **B-11** 实现 `clone() → Board`
  - 创建新 Board 对象
  - 深拷贝 `grid`、`amazon_positions`、`move_history`（或直接复制整个 `UndoInfo` 列表）
  - 复制 `current_player`、`zobrist_hash`

- [ ] **B-12** 实现 `reset() → None`
  - 重新执行 `__init__` 的逻辑
  - `move_history = []`

## 测试任务

- [ ] **T-01** 创建 `tests/test_board.py`
- [ ] **T-02** 测试初始棋盘
  - `get_piece_at(0,3)` 应为 `BLACK`（d10）
  - `get_piece_at(9,3)` 应为 `WHITE`（d1）
  - `get_current_player()` 应为 `WHITE`
  - `len(get_amazon_positions(WHITE))` 应为 4
- [ ] **T-03** 测试 `make_move` 后状态
  - 构造走法 `d1→d5→g7`：`Move((9,3),(5,3),(3,6))`
  - `make_move` 后：d1 为空，d5 有白子，g7 为烧毁格
  - `current_player` 变为 `BLACK`
- [ ] **T-04** 测试 `undo_move` 恢复
  - `make_move` → `undo_move` → 棋盘状态与执行前完全一致
  - 包括 `current_player`、`amazon_positions`、`zobrist_hash`
- [ ] **T-05** 测试 `make_move` + `undo_move` 往返 10 次随机走法
- [ ] **T-06** 测试 `clone` 独立性
  - `b2 = b1.clone()` → `b2.make_move(...)` → `b1` 不受影响
- [ ] **T-07** 测试 `reset()` → 棋盘回到初始布局
- [ ] **T-08** 运行 `python -m pytest tests/test_board.py -v`，全部通过

## 完成标准
- [ ] 初始布局 8 枚棋子在正确位置
- [ ] `make_move` / `undo_move` 正确修改和恢复所有状态
- [ ] `clone` 产生完全独立的副本
- [ ] 全部测试通过
