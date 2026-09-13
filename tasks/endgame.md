# endgame 任务清单

> **模块**：`src/endgame/tablebase.py` + `src/endgame/retrograde.py`  
> **依赖**：`game-board`, `game-rules`  
> **预估工时**：2 小时  

## 实现任务

### 数据结构

- [ ] **EG-01** 创建 `src/endgame/__init__.py`（空文件）
- [ ] **EG-02** 创建 `src/endgame/data/` 目录

- [ ] **EG-03** 创建 `src/endgame/tablebase.py`
- [ ] **EG-04** 定义 `EndgameResult` 数据类
  ```python
  result: int    # WIN=1 / LOSE=-1
  distance: int  # 距终局步数
  ```

### Tablebase（查询器）

- [ ] **EG-05** 实现 `Tablebase.__init__(data_path=None)`
  - 若 `data_path` 为 None，跳过加载
  - 否则用 `pickle.load()` 加载 `.pkl` 文件

- [ ] **EG-06** 实现 `Tablebase.query(board_hash: int) → EndgameResult | None`
  - 在内部字典中查找 `board_hash`
  - 命中返回 `EndgameResult`，否则 None

- [ ] **EG-07** 实现 `Tablebase.is_available() → bool`

### RetrogradeGenerator（回溯分析生成器）

- [ ] **EG-08** 创建 `src/endgame/retrograde.py`
- [ ] **EG-09** 实现 `RetrogradeGenerator.__init__(max_pieces=3, max_empty=15)`

- [ ] **EG-10** 实现 `estimate_state_space() → int`
  - 粗略估算状态数量（用于判断计算可行性）
  - 公式：C(n, pieces) × 排列方式，大致量级即可

- [ ] **EG-11** 实现 `generate(output_path: str) → dict`
  1. 枚举所有满足条件的子局面（棋子组合 + 空格/烧毁格分配）
  2. 对每个局面计算 Zobrist 哈希 → 存储
  3. 标记所有终局状态（某方无合法走法）→ result=LOSE
  4. **反向迭代**：
     - 对每个未标记局面 S：
       - 若从 S 可走到已标记为 LOSE 的局面 → S 标记为 WIN（distance = min + 1）
       - 若从 S 所有走法都走向已标记为 WIN 的局面 → S 标记为 LOSE（distance = max + 1）
     - 重复直到无新标记（不动点）
  5. 序列化保存为 `.pkl`
  6. 打印统计信息（总局面数、WIN/LOSE 分布）

## 测试任务

- [ ] **T-01** 创建 `tests/test_endgame.py`
- [ ] **T-02** 用极小范围（2 子、4 空格）运行 `generate()`，验证输出非空
- [ ] **T-03** 用输出的 `.pkl` 加载 `Tablebase`，验证加载成功
- [ ] **T-04** 对已知必胜/必败局面查询 `Tablebase.query()`
- [ ] **T-05** 对不在范围内的局面查询返回 None
- [ ] **T-06** 测试 `estimate_state_space()` 返回值合理（数量级正确）
- [ ] **T-07** 运行 `python -m pytest tests/test_endgame.py -v`，全部通过

## 完成标准
- [ ] 极小范围回溯分析产生正确结果
- [ ] Tablebase 正确加载和查询
- [ ] 全部测试通过

> ⚠️ 残局库预计算为**可选任务**。若初赛前时间不足，可先完成 Tablebase 查询接口（EG-05~07），`generate()` 留待初赛后执行。
