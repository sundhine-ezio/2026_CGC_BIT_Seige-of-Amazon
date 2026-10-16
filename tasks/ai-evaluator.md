# ai-evaluator 任务清单

> **模块**：`src/ai/evaluator.py`  
> **依赖**：`game-board`, `game-rules`  
> **预估工时**：1.5 小时  

## 实现任务

- [ ] **EV-01** 创建 `src/ai/evaluator.py`，导入 `random`, `Board`, `Rules`

- [ ] **EV-02** 实现 `_random_playout(board: Board) → int`（模块级函数）
  1. `while not Rules.is_game_over(board):`
  2. `moves = Rules.get_legal_moves(board)`
  3. `if not moves: break`
  4. `board.make_move(random.choice(moves))`
  5. 返回 `Rules.get_winner(board)`

- [ ] **EV-03** 实现 `MonteCarloEvaluator.__init__(num_simulations=100)`
  - 保存 `num_simulations`
  - 可选：保存随机种子以便复现

- [ ] **EV-04** 实现 `MonteCarloEvaluator.evaluate(board: Board) → float`
  1. 记录当前玩家 `player = board.get_current_player()`
  2. 循环 `num_simulations` 次：
     - `sim_board = board.clone()`
     - `result = _random_playout(sim_board)`
     - 若 `result == player`，`wins += 1`
  3. 返回 `wins / num_simulations`

- [ ] **EV-05** 实现 `set_num_simulations(n: int)`：动态调整模拟次数

## 测试任务

- [ ] **T-01** 创建 `tests/test_evaluator.py`
- [ ] **T-02** 测试评估值范围：任意局面下 `0.0 <= evaluate(board) <= 1.0`
- [ ] **T-03** 测试必胜局面（对手无合法走法）→ `evaluate()` 返回 `1.0`
- [ ] **T-04** 测试必败局面（己方无合法走法）→ `evaluate()` 返回 `0.0`
- [ ] **T-05** 测试确定性：固定 `random.seed(42)`，两次 `evaluate` 同一局面应返回相同值
- [ ] **T-06** 测试 `clone` 未被破坏：`evaluate` 后原 board 状态不变
- [ ] **T-07** 测试 `set_num_simulations` 生效：`n=10` 和 `n=100` 的评估时间差异明显
- [ ] **T-08** 运行 `python -m pytest tests/test_evaluator.py -v`，全部通过

## 完成标准
- [ ] 必败/必胜局面评估准确
- [ ] 中间局面评估值在合法范围内
- [ ] `evaluate()` 不修改传入的 Board
- [ ] 全部测试通过
