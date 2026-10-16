# game-move 任务清单

> **模块**：`src/game/move.py`  
> **依赖**：`game-constants`  
> **预估工时**：1 小时  

## 实现任务

- [ ] **M-01** 创建 `src/game/move.py`，导入 `dataclass` 和 `constants`
- [ ] **M-02** 定义 `Move` 数据类（`@dataclass(frozen=True)`）
  ```python
  from_pos: tuple[int, int]
  to_pos: tuple[int, int]
  arrow_pos: tuple[int, int]
  ```
- [ ] **M-03** 实现 `Move.from_string(s: str) → Move`
  - 按 `"-"` 拆分字符串为三段
  - 每段调用 `algebraic_to_pos()` 解析
  - 若段数 ≠ 3 或坐标非法，抛出 `ValueError`
- [ ] **M-04** 实现 `Move.to_string() → str`
  - 格式：`"起点-移动终点-箭落点"`，每段用 `pos_to_algebraic()`
  - 例：`Move((9,3),(5,3),(3,6)) → "d1-d5-g7"`
- [ ] **M-05** 实现 `Move.__repr__() → str`：直接调用 `to_string()`
- [ ] **M-06** 实现 `Move.__eq__()` 比较方法（`frozen=True` 已自动生成，验证即可）

## 测试任务

- [ ] **T-01** 创建 `tests/test_move.py`
- [ ] **T-02** 测试 `from_string → to_string` 往返
  - 输入 `"d10-d5-g7"`，输出应相同
  - 输入 `"a7-a10-d7"`，输出应相同
- [ ] **T-03** 测试 `from_string` 非法输入
  - `""` → `ValueError`
  - `"a4-d4"`（少一段）→ `ValueError`
  - `"a4-d4-g7-x1"`（多一段）→ `ValueError`
  - `"z4-d4-g7"`（列越界）→ `ValueError`
- [ ] **T-04** 测试边界坐标
  - `"a1-j10-a1"` → 各坐标正确
  - `"j10-a1-j10"` → 各坐标正确
- [ ] **T-05** 测试 `frozen=True`：尝试修改 Move 的属性应抛 `FrozenInstanceError`
- [ ] **T-06** 测试两个相同 Move 的 `==` 比较为 True
- [ ] **T-07** 运行 `python -m pytest tests/test_move.py -v`，全部通过

## 完成标准
- [ ] `Move.from_string()` 和 `Move.to_string()` 互为逆操作
- [ ] 非法输入均抛出 `ValueError`
- [ ] Move 对象不可变（frozen）
- [ ] 全部测试通过
