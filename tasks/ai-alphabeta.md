# ai-alphabeta 任务清单

> **模块**：`src/ai/alphabeta.py`  
> **依赖**：`ai-engine`, `ai-evaluator`, `ai-transposition`, `game-board`, `game-rules`  
> **预估工时**：3 小时  

## 实现任务

### 数据结构

- [ ] **AB-01** 创建 `src/ai/alphabeta.py`
- [ ] **AB-02** 定义 `SearchResult` 数据类：`move: Move | None`, `value: float`
- [ ] **AB-03** 定义 `SearchStats` 数据类：`nodes_visited`, `leaves_evaluated`, `transposition_hits`, `depth_reached`, `time_elapsed_ms`

### 核心搜索

- [ ] **AB-04** 实现 `AlphaBetaEngine.__init__()`
  ```python
  def __init__(self, evaluator, max_depth=5, time_budget_ms=15000.0,
               use_transposition_table=True, use_iterative_deepening=True,
               use_move_ordering=False, transposition_table_size=2**20):
  ```
  - 保存所有配置参数
  - 若启用置换表，实例化 `TranspositionTable`
  - 初始化 `opening_book = None`, `tablebase = None`
  - 初始化 `stats = SearchStats()`

- [ ] **AB-05** 实现 `_check_timeout(start_time, time_budget_ms) → bool`
  - `time.time() - start_time > time_budget_ms * 0.8`

- [ ] **AB-06** 实现 `_alpha_beta(board, depth, alpha, beta, is_maximizing, start_time, time_budget_ms) → float`
  1. **超时检查**：若超时，抛出 `TimeoutError`（自定义异常）
  2. **终局检查**：`Rules.is_game_over(board)` → 返回胜负值（当前玩家视角，1.0 或 0.0）
  3. **深度耗尽** (`depth == 0`)：
     - 查询置换表（若命中 → 返回）
     - 查询残局库（若命中 → 返回精确值）
     - 调用 `evaluator.evaluate(board)` → 返回
  4. **走法生成**：`Rules.get_legal_moves(board)`
  5. **Alpha-Beta 递归**：
     - 若 `is_maximizing`：value=max, alpha 更新, alpha≥beta 剪枝
     - 若 `not is_maximizing`：value=min, beta 更新, alpha≥beta 剪枝
  6. 走法执行用 `board.make_move`/`board.undo_move`

- [ ] **AB-07** 实现 `fixed_depth_search(board, depth, start_time=None, time_budget_ms=float('inf')) → SearchResult`
  - 对根节点的每个合法走法，调用 `_alpha_beta(board, depth-1, alpha, beta, False, ...)`
  - 跟踪最优走法和最优值
  - 若超时则返回当前已找到的最优结果
  - 返回 `SearchResult(best_move, best_value)`

- [ ] **AB-08** 实现 `iterative_deepening_search(board, start_time, time_budget_ms) → SearchResult`
  1. 保底走法 = 第一个合法走法
  2. `for depth = 1 to max_depth`:
     - 若超时 → break
     - 调用 `fixed_depth_search(board, depth, start_time, time_budget_ms)`
     - 若完成（未超时），更新 best_result
     - 若 `best_result.value > 0.99`（必胜），提前终止
  3. 记录 `stats.depth_reached`

### 对外接口

- [ ] **AB-09** 实现 `search(board, time_budget_ms) → Move`
  1. 若启用开局库且命中 → 直接返回走法
  2. 若启用迭代加深 → `iterative_deepening_search()`
  3. 否则 → `fixed_depth_search(board, max_depth)`
  4. 返回 `result.move`

- [ ] **AB-10** 实现 `get_name() → str`

- [ ] **AB-11** 实现 `reset()`：清空置换表、重置统计

- [ ] **AB-12** 实现 `get_stats() → SearchStats`

- [ ] **AB-13** 实现 `set_opening_book(book)` / `set_endgame_tablebase(tablebase)`

### TimeoutError

- [ ] **AB-14** 定义 `class TimeoutError(Exception): pass`
- [ ] **AB-15** 在 `fixed_depth_search` 和 `iterative_deepening_search` 中捕获 `TimeoutError`

## 测试任务

- [ ] **T-01** 创建 `tests/test_alphabeta.py`
- [ ] **T-02** 测试简单一步杀：构造一方亚马逊可一步封死对方的局面 → depth=1 应找到该走法
- [ ] **T-03** 测试必败局面：己方无合法走法 → search 返回保底走法（不崩溃）
- [ ] **T-04** 测试超时返回：设定 0.001 秒超时 → 仍返回合法走法
- [ ] **T-05** 测试迭代加深：depth=2 的结果不差于 depth=1 的结果（统计检验）
- [ ] **T-06** 测试置换表加速：同一局面两次搜索，第二次明显更快
- [ ] **T-07** 测试 `reset()` 后置换表为空
- [ ] **T-08** 创建 `tests/test_alphabeta_integration.py`
- [ ] **T-09** AI vs Random 100 局：AlphaBeta(depth=3) 胜率 > 90%
- [ ] **T-10** AI vs AI 完整对弈：每步走法均为合法走法，终局胜者判定正确
- [ ] **T-11** 运行 `python -m pytest tests/test_alphabeta*.py -v`，全部通过

## 完成标准
- [ ] 搜索始终返回合法走法（即使超时）
- [ ] 一步杀局面能正确识别
- [ ] 迭代加深深度递增
- [ ] 对随机 AI 胜率 > 90%
- [ ] 全部测试通过
