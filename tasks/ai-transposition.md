# ai-transposition 任务清单

> **模块**：`src/ai/transposition.py`  
> **依赖**：`game-board`, `game-constants`  
> **预估工时**：1.5 小时  

## 实现任务

- [ ] **TR-01** 创建 `src/ai/transposition.py`，导入 `random`, `Move`, `constants`

### ZobristHasher

- [ ] **TR-02** 实现 `ZobristHasher` 类（类级别单例模式）
  ```python
  _table: dict[tuple[int,int,int], int]  # (piece, row, col) → 64-bit hash
  _side_to_move_rand: int
  ```

- [ ] **TR-03** 实现 `ZobristHasher.init(seed=42)`
  - 用 `random.Random(seed)` 生成确定性随机数
  - 对每种 `piece ∈ (WHITE, BLACK, BURNED)` × `row 0..9` × `col 0..9` 生成 64 位随机数（共 300 个）
  - 生成 1 个 `_side_to_move_rand`

- [ ] **TR-04** 实现 `ZobristHasher.hash_piece(piece, row, col) → int`
  - 返回 `_table[(piece, row, col)]`

- [ ] **TR-05** 实现 `ZobristHasher.hash_side_to_move() → int`

- [ ] **TR-06** **集成到 Board**：在 `Board.__init__` 中调用 `ZobristHasher.init()`（若未初始化），修改 `Board.__init__` 计算初始 Zobrist 哈希：
  ```python
  zobrist_hash = ZobristHasher.hash_side_to_move()
  for pos in INITIAL_BLACK_POSITIONS:
      zobrist_hash ^= ZobristHasher.hash_piece(BLACK, *pos)
  for pos in INITIAL_WHITE_POSITIONS:
      zobrist_hash ^= ZobristHasher.hash_piece(WHITE, *pos)
  ```

- [ ] **TR-07** 修改 `Board.make_move` 中的哈希更新逻辑（增量 XOR）
  ```python
  # 移除 from_pos 的棋子
  hash ^= hash_piece(piece, *from_pos)
  # 放置 to_pos 的棋子
  hash ^= hash_piece(piece, *to_pos)
  # 放置烧毁格
  hash ^= hash_piece(BURNED, *arrow_pos)
  # 切换走子方
  hash ^= hash_side_to_move()
  ```

- [ ] **TR-08** 修改 `Board.undo_move` 恢复旧哈希值（从 `UndoInfo.prev_hash` 直接恢复）

### TranspositionTable

- [ ] **TR-09** 定义 `TranspositionEntry` 数据类
  ```python
  hash_key: int
  depth: int
  value: float
  flag: int    # EXACT=0, LOWER_BOUND=1, UPPER_BOUND=2
  best_move: Move | None
  ```

- [ ] **TR-10** 实现 `TranspositionTable.__init__(size=2**20)`
  - `table = [None] * size`（大小为 2 的幂便于位运算取模）

- [ ] **TR-11** 实现 `TranspositionTable._index(hash_key) → int`
  - `hash_key & (size - 1)` 或 `hash_key % size`

- [ ] **TR-12** 实现 `TranspositionTable.lookup(hash_key) → TranspositionEntry | None`
  - 取模定位 → 检查槽位非空且 `entry.hash_key == hash_key` → 命中返回，否则 None

- [ ] **TR-13** 实现 `TranspositionTable.store(hash_key, depth, value, flag, best_move=None)`
  - 替换策略：槽位为空 **或** 新 depth ≥ 旧 depth → 覆盖

- [ ] **TR-14** 实现 `TranspositionTable.clear()`

## 测试任务

- [ ] **T-01** 创建 `tests/test_transposition.py`
- [ ] **T-02** 测试 Zobrist 哈希：同一局面两次计算哈希值相同
- [ ] **T-03** 测试 Zobrist 哈希：不同局面哈希值不同（统计 1000 个随机局面无明显碰撞）
- [ ] **T-04** 测试增量更新：`init_hash == 逐步 XOR 计算结果`
- [ ] **T-05** 测试置换表 `store → lookup` 往返正确
- [ ] **T-06** 测试 `clear` 后 `lookup` 返回 None
- [ ] **T-07** 测试置换表深度优先替换策略
- [ ] **T-08** 运行 `python -m pytest tests/test_transposition.py -v`，全部通过

## 完成标准
- [ ] Zobrist 哈希增量更新正确（性能优化可后续验证）
- [ ] 置换表存取正确、替换策略正确
- [ ] 全部测试通过
