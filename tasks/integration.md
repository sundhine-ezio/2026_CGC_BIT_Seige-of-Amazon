# integration 任务清单

> **模块**：集成测试与端到端验证  
> **依赖**：全部模块  
> **预估工时**：2 小时  

## 实现任务

### 冒烟测试（阶段 1.1 完成后执行）

- [ ] **IT-01** 运行 `python -m pytest tests/ -v`，确保全部单元测试通过
- [ ] **IT-02** 启动 `python main.py --mode console --depth 1`，程序能启动并完成一步走法

### 基准对弈测试（阶段 1.1 完成后执行）

- [ ] **IT-03** 编写 `tests/test_match.py`：批量对弈测试脚本
  ```python
  def run_match(engine_white, engine_black, num_games, time_budget_ms):
      """运行 num_games 局，返回 (white_wins, black_wins, draws, avg_moves)"""
  ```

- [ ] **IT-04** **AlphaBeta (depth=3) vs Random**：100 局
  - 预期胜率 > 90%
  - 记录并验证每条走法均为合法走法

- [ ] **IT-05** **AlphaBeta (depth=3) vs AlphaBeta (depth=1)**：20 局
  - 预期 depth=3 方胜率 > 80%

- [ ] **IT-06** **AlphaBeta (depth=4) vs AlphaBeta (depth=3)**：10 局
  - 预期 depth=4 方胜率 ≥ 50%（更深不一定绝对优势，但不应明显更差）

### 正确性验证

- [ ] **IT-07** 验证完整对弈不崩溃：AI vs AI 运行 10 局，每局从头到尾
- [ ] **IT-08** 验证终局裁决正确：每局 `get_winner()` 的返回值与实际终局逻辑一致

### 性能测试

- [ ] **IT-09** 计时测试：depth=3 在初始棋盘上的单步搜索时间
  - 记录 avg / max / min 三档
  - 确保不超过 30 秒（为 15 分钟总时限留足余量）

- [ ] **IT-10** 全局面测试：运行 3 局完整对弈，记录每步搜索时间和总时间
  - 确保 15 分钟内能完成一整局（通常一局约 40-80 步）
  - 若超时，建议降低 `num_simulations` 或 `max_depth`

### 比赛模式通信验证

- [ ] **IT-11** 编写 `tests/test_competition_protocol.py`
- [ ] **IT-12** 模拟对手发 "d1-d5-g8" → 验证程序返回合法走法
- [ ] **IT-13** 模拟对手连续发 10 步 → 验证程序每步正确响应
- [ ] **IT-14** 测试非法输入：对手发 "invalid" → 程序抛出 ValueError（符合严格模式）

### GUI 手工验证（阶段 1.2 完成后执行）

- [ ] **IT-15** 启动 GUI，确认窗口正常显示
- [ ] **IT-16** 人类执白 vs AI 执黑：完成一局完整对弈
- [ ] **IT-17** AI vs AI 观战模式：自动播放一局完整对弈

## 完成标准
- [ ] 全部单元测试通过
- [ ] AlphaBeta depth=3 对 Random 胜率 > 90%
- [ ] 一局完整对弈在 15 分钟内完成
- [ ] 三种运行模式均可正常使用
- [ ] 比赛模式通信协议正确
