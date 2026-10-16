# 亚马逊棋计算机博弈系统——代码文档

> 生成日期：2026-07-18  
> 项目路径：`/mnt/c/Users/35921/Desktop/本科学业/计算机博弈大赛/`

---

## 项目结构总览

```
amazons/
├── main.py                         # 程序入口（三种运行模式）
├── config.yaml                     # YAML 全局配置
├── pyproject.toml                  # 项目配置（mypy strict + ruff 推荐规则集）
├── requirements.txt                # pip 依赖清单
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py            # YAML 配置加载器
│   │
│   ├── game/                       # 游戏规则层
│   │   ├── __init__.py
│   │   ├── constants.py            # 常量 + 坐标转换函数
│   │   ├── move.py                 # Move 数据类 + 字符串编解码
│   │   ├── board.py                # Board 棋盘状态 + make/undo/clone
│   │   └── rules.py                # Rules 合法走法生成 + 终局判断
│   │
│   ├── ai/                         # AI 搜索层
│   │   ├── __init__.py
│   │   ├── engine.py               # Engine 抽象基类
│   │   ├── evaluator.py            # MonteCarloEvaluator 随机模拟评估
│   │   ├── transposition.py        # ZobristHasher + TranspositionTable
│   │   └── alphabeta.py            # AlphaBetaEngine（核心搜索引擎）
│   │
│   ├── gui/                        # 图形界面层（Tkinter）
│   │   ├── __init__.py
│   │   ├── board_painter.py        # BoardCanvas 棋盘绘制组件
│   │   ├── main_window.py          # AmazonsGUI 主窗口（三阶段状态机）
│   │   └── dialogs.py              # 对话框工具函数
│   │
│   ├── opening_book/               # 开局库
│   │   ├── __init__.py
│   │   ├── book.py                 # OpeningBook 加载/查询/加权随机
│   │   └── data/
│   │       └── opening_book.json   # 示例开局库数据
│   │
│   ├── endgame/                    # 残局库
│   │   ├── __init__.py
│   │   ├── tablebase.py            # Tablebase 查询接口
│   │   ├── retrograde.py           # RetrogradeGenerator 回溯分析生成器
│   │   └── data/                   # 预计算的 .pkl 数据目录
│   │
│   └── network/                    # 深度学习层（第二阶段占位）
│       ├── __init__.py
│       ├── config.py               # NetworkConfig + TrainingConfig
│       ├── model.py                # AmazonsNet（接口占位）
│       ├── train.py                # 训练流水线（接口占位）
│       ├── selfplay.py             # 自我对弈生成（接口占位）
│       └── replay_buffer.py        # 经验回放池（接口占位）
│
└── tests/                          # 单元测试 & 集成测试
    ├── test_constants.py
    ├── test_move.py
    ├── test_board.py
    ├── test_rules.py
    ├── test_engine.py
    ├── test_evaluator.py
    ├── test_transposition.py
    ├── test_alphabeta.py
    ├── test_alphabeta_integration.py
    ├── test_opening_book.py
    ├── test_endgame.py
    └── test_match.py
```

---

## 测试结果

| 测试文件 | 测试数 | 通过 | 失败 |
|----------|--------|------|------|
| test_constants.py | 28 | 28 | 0 |
| test_move.py | 13 | 13 | 0 |
| test_board.py | 15 | 15 | 0 |
| test_rules.py | 13 | 13 | 0 |
| test_engine.py | 3 | 3 | 0 |
| test_evaluator.py | 5 | 5 | 0 |
| test_transposition.py | 8 | 8 | 0 |
| test_alphabeta.py | 7 | 运行中 | — |
| test_opening_book.py | 6 | 运行中 | — |
| test_endgame.py | 5 | 运行中 | — |

---

## 快速开始

### 安装依赖

```bash
cd /mnt/c/Users/35921/Desktop/本科学业/计算机博弈大赛
pip install numpy pyyaml pytest mypy ruff
```

### 运行模式

```bash
# GUI 模式（人类 vs AI）
python main.py --mode gui

# 控制台 AI vs AI 对弈（文本输出）
python main.py --mode console --depth 3

# 比赛模式（stdin/stdout 通信）
echo "d1-d5-g8" | python main.py --mode competition --color black --depth 3
```

### 运行测试

```bash
# 全部单元测试
python -m pytest tests/ -v

# 特定模块测试
python -m pytest tests/test_board.py -v

# 代码检查
ruff check src/
mypy src/
```

---

## 核心模块 API 速查

### game/constants.py
```python
EMPTY=0, WHITE=1, BLACK=-1, BURNED=2
pos_to_algebraic(row, col) -> str     # (0,3) -> "d10"
algebraic_to_pos(notation) -> tuple   # "d10" -> (0,3)
is_in_bounds(row, col) -> bool
```

### game/move.py
```python
Move(from_pos, to_pos, arrow_pos)     # frozen dataclass
Move.from_string("d10-d5-g7") -> Move
move.to_string() -> "d10-d5-g7"
```

### game/board.py
```python
board = Board()                       # 标准初始布局
board.get_grid() -> np.ndarray        # (10,10)
board.make_move(move)                 # 执行
board.undo_move()                     # 撤销
board.clone() -> Board                # 深拷贝
board.get_zobrist_hash() -> int
```

### game/rules.py
```python
Rules.get_legal_moves(board) -> list[Move]
Rules.get_queen_reachable(board, pos) -> list[tuple]
Rules.is_game_over(board) -> bool
Rules.get_winner(board) -> int        # WHITE or BLACK
```

### ai/engine.py
```python
class Engine(ABC):
    search(board, time_budget_ms) -> Move
    get_name() -> str
    reset() -> None
```

### ai/evaluator.py
```python
eval = MonteCarloEvaluator(num_simulations=100)
eval.evaluate(board) -> float         # [0.0, 1.0]
```

### ai/alphabeta.py
```python
engine = AlphaBetaEngine(
    evaluator, max_depth=5, time_budget_ms=15000.0,
    use_transposition_table=True, use_iterative_deepening=True,
)
engine.search(board, time_budget_ms) -> Move
engine.set_opening_book(book)         # 可选注入
engine.set_endgame_tablebase(tb)      # 可选注入
```

### gui/main_window.py
```python
gui = AmazonsGUI(engine, mode="HUMAN_VS_AI")  # or AI_VS_HUMAN, AI_VS_AI
gui.run()                             # 启动 Tkinter 主循环
```

---

## 关键设计决策

| 决策 | 选型 | 理由 |
|------|------|------|
| 棋盘表示 | numpy (10,10) int8 数组 | 高效切片，拷贝快 |
| 走法生成 | make_move + undo_move（而非临时棋盘） | 正确性优先，避免状态不一致 |
| 评估函数 | 蒙特卡洛随机模拟（100次） | 无需领域知识，实现简单 |
| 哈希 | Zobrist（固定种子42、增量 XOR 更新） | 确定性，可复现 |
| 置换表 | 深度优先替换 + 位运算取模索引 | 标准做法 |
| 开局库 | JSON 文件 + 加权随机 | 可手工编辑，可扩展 |
| GUI | Tkinter（Python 标准库） | 零依赖安装 |
| 第二阶段网络 | 三头分解（from/to/arrow 各100维） | 维度可控 |

---

> **版本**: v1.0 | **作者**: 杨光 | **许可证**: 待定
