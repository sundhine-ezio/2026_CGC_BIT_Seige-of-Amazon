# game-constants 任务清单

> **模块**：`src/game/constants.py`  
> **依赖**：无  
> **预估工时**：0.5 小时  

## 实现任务

- [ ] **C-01** 创建 `src/game/constants.py` 文件，顶部写模块 docstring
- [ ] **C-02** 定义棋盘尺寸常量 `BOARD_SIZE = 10`
- [ ] **C-03** 定义格位编码常量
  ```python
  EMPTY = 0    # 空格
  WHITE = 1    # 白方亚马逊
  BLACK = -1   # 黑方亚马逊
  BURNED = 2   # 烧毁格
  ```
- [ ] **C-04** 定义 `QUEEN_DIRECTIONS`：8 个方向的 (dr, dc) 元组列表
- [ ] **C-05** 定义 `COL_LABELS = "abcdefghij"`、`ROW_LABELS`（注意：行号字符串包含 "10"，解析时需特殊处理）
- [ ] **C-06** 定义初始布局常量 `INITIAL_BLACK_POSITIONS`：`[(0,3), (0,6), (3,0), (3,9)]`（d10, g10, a7, j7）
- [ ] **C-07** 定义初始布局常量 `INITIAL_WHITE_POSITIONS`：`[(6,0), (6,9), (9,3), (9,6)]`（a4, j4, d1, g1）
- [ ] **C-08** 实现 `pos_to_algebraic(row, col) → str`：row=0→"10", row=9→"1", col→"a"-"j"
- [ ] **C-09** 实现 `algebraic_to_pos(notation) → tuple[int, int]`：含 "10" 行号的特殊解析逻辑
- [ ] **C-10** 实现 `is_in_bounds(row, col) → bool`：判断是否在 0..9 范围内
- [ ] **C-11** 定义胜负常量 `WIN=1, LOSE=-1, DRAW=0`

## 测试任务

- [ ] **T-01** 创建 `tests/test_constants.py`
- [ ] **T-02** 测试 `pos_to_algebraic`：`(0,0)→"a10"`, `(9,9)→"j1"`, `(3,0)→"a7"`
- [ ] **T-03** 测试 `algebraic_to_pos`：`"a10"→(0,0)`, `"j1"→(9,9)`, `"d1"→(9,3)`, `"g10"→(0,6)`
- [ ] **T-04** 测试往返一致性：`algebraic_to_pos(pos_to_algebraic(r,c)) == (r,c)` 对所有初始棋子位置
- [ ] **T-05** 测试 `is_in_bounds`：边界值 (0,0)→True, (9,9)→True, (-1,0)→False, (10,5)→False
- [ ] **T-06** 运行 `python -m pytest tests/test_constants.py -v`，全部通过

## 完成标准
- [ ] 所有常量可被 `from src.game.constants import *` 导入
- [ ] 坐标转换往返无误差
- [ ] 全部测试通过
