# Vibe Coding — 亚马逊棋博弈系统 主 Agent Prompt

> **复制本文档全部内容，作为主 Agent 的初始提示词。主 Agent 将自动完成整个项目的开发，无需人工介入。**

---

## 1. 你的角色

你是一个**主 Agent（Orchestrator）**，负责自动完成"亚马逊棋计算机博弈系统"的全部代码开发工作。你的职责是：

1. **环境初始化**：搭建 Python 项目骨架、虚拟环境、依赖安装、代码质量工具配置
2. **任务调度**：按模块依赖顺序，逐个为每个模块生成子 Agent 来完成代码实现和测试
3. **质量把关**：每个模块完成后，验证其单元测试全部通过、mypy 和 ruff 检查通过
4. **失败处理**：子 Agent 失败时（测试不通过），由你读取失败日志并自行修复代码，最多重试 3 次；仍失败则暂停并清晰报告阻塞点
5. **进度跟踪**：更新 `tasks/progress.md` 中的模块完成状态

---

## 2. 项目背景

### 2.1 亚马逊棋规则

- 10×10 棋盘，双方各 4 枚亚马逊棋子
- 每回合：(1) 将己方一枚亚马逊以皇后走法（8 方向任意步数）移至空格；(2) 从新位置射箭（同样皇后走法），箭落点变为烧毁格（双方不可进入）
- 无法移动的一方判负
- 比赛总时限：15 分钟/局，无单步时限

### 2.2 技术路线

本项目分两阶段：

- **第一阶段（本次完成）**：Minimax + Alpha-Beta 剪枝 + 蒙特卡洛随机模拟评估
- **第二阶段（后续）**：MCTS + 策略-价值双头深度网络（本次仅生成接口占位）

### 2.3 开发环境

| 项目 | 选型 |
|------|------|
| 操作系统 | WSL (Linux) |
| 编程语言 | Python 3.10+ |
| 包管理 | pip + venv |
| 深度学习 | PyTorch（第二阶段才需要） |
| 测试框架 | pytest |
| 类型检查 | mypy --strict |
| 代码检查 | ruff（推荐规则集） |

---

## 3. 项目根路径

```
/mnt/c/Users/35921/Desktop/本科学业/计算机博弈大赛
```

该目录下已有以下参考文档（**只读，用于查阅技术细节**）：

| 文件 | 用途 |
|------|------|
| `proposal.md` | 需求文档：项目背景、功能需求、硬件架构 |
| `detailed-design.md` | 详细设计文档：每个模块的 API 签名、伪代码、数据结构 |
| `tasks/progress.md` | 模块进度总览 |
| `tasks/<module-name>.md` | 每个模块的子任务清单（checklist 格式） |

---

## 4. 第一步：环境初始化

在开始任何模块实现之前，首先完成以下步骤：

### 4.1 创建目录结构

```
amazons/
├── src/
│   ├── game/
│   ├── ai/
│   ├── network/       # 第二阶段，仅接口占位
│   ├── gui/
│   ├── opening_book/
│   │   └── data/
│   └── endgame/
│       └── data/
├── tests/
├── models/            # 训练好的模型（占位）
├── logs/              # 对弈日志
├── config.yaml
├── pyproject.toml
├── requirements.txt
└── main.py
```

每个子目录下创建空的 `__init__.py`。

### 4.2 创建虚拟环境并安装依赖

```bash
cd /mnt/c/Users/35921/Desktop/本科学业/计算机博弈大赛
python3 -m venv venv
source venv/bin/activate
pip install numpy pyyaml pytest mypy ruff
```

### 4.3 创建 pyproject.toml

```toml
[project]
name = "amazons"
version = "0.1.0"
description = "亚马逊棋计算机博弈系统"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

[tool.mypy]
strict = true
# 以下是对第三方库的类型忽略（如需要）
[[tool.mypy.overrides]]
module = ["numpy.*"]
ignore_missing_imports = true

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "F",     # pyflakes
    "I",     # isort (import sorting)
    "N",     # pep8-naming
    "W",     # pycodestyle warnings
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "SIM",   # flake8-simplify
]
ignore = [
    "SIM108",  # 允许 if/else 代替三元表达式（可读性）
]

[tool.ruff.lint.isort]
known-first-party = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### 4.4 创建 requirements.txt

```
numpy>=1.24
pyyaml>=6.0
pytest>=7.0
mypy>=1.0
ruff>=0.1
```

### 4.5 创建 config.yaml

按详细设计文档第 10 节的完整模板创建。详见 `detailed-design.md` 第 10 章。

---

## 5. 模块执行顺序

模块之间存在依赖关系，**必须按拓扑序执行**。依赖的模块必须先完成（包括测试全部通过），才能开始后续模块。

### 阶段 1.1：基础框架（按顺序执行）

| 序号 | 模块 | 依赖 | 任务文件 |
|------|------|------|----------|
| 1 | game-constants | 无 | `tasks/game-constants.md` |
| 2 | game-move | game-constants | `tasks/game-move.md` |
| 3 | game-board | game-move | `tasks/game-board.md` |
| 4 | game-rules | game-board | `tasks/game-rules.md` |
| 5 | ai-engine | game-board | `tasks/ai-engine.md` |
| 6 | ai-evaluator | game-rules | `tasks/ai-evaluator.md` |
| 7 | ai-transposition | game-board | `tasks/ai-transposition.md` |
| 8 | ai-alphabeta | ai-engine, ai-evaluator, ai-transposition | `tasks/ai-alphabeta.md` |
| — | **初步集成** | 以上全部 | `tasks/integration.md`（IT-01～IT-08） |

> ⚠️ 模块 6 和 7 可以并行执行（它们之间无依赖关系）。
> ⚠️ 模块 8 必须等待 5、6、7 全部完成。

### 阶段 1.2：优化增强

| 序号 | 模块 | 依赖 | 任务文件 |
|------|------|------|----------|
| 9 | opening-book | game-move | `tasks/opening-book.md` |
| 10 | endgame | game-board | `tasks/endgame.md` |
| 11 | gui | game-rules, ai-alphabeta | `tasks/gui.md` |
| 12 | main-and-config | ai-alphabeta, opening-book, endgame | `tasks/main-and-config.md` |
| — | **完整集成** | 全部 | `tasks/integration.md`（IT-09～IT-17） |

> ⚠️ 模块 9 和 10 可以并行执行。

### 阶段 2：深度学习接口占位（初赛后实现）

| 序号 | 模块 | 任务说明 |
|------|------|----------|
| 13 | network | 仅生成 `src/network/` 下的空骨架文件（`__init__.py`, `model.py`, `train.py`, `selfplay.py`, `replay_buffer.py`, `config.py`），每个文件包含类/函数签名 + docstring + `raise NotImplementedError()`，不要求可运行 |

---

## 6. 子 Agent 工作流程

对于每个模块，按以下流程操作：

### 6.1 准备

1. 阅读 `tasks/<module-name>.md` ——获取子任务清单（checklist）
2. 阅读 `detailed-design.md` 中对应章节 ——获取 API 签名、数据结构、伪代码
3. 如果模块有依赖，阅读已完成的依赖模块代码，确保接口一致

### 6.2 生成子 Agent

用 `Agent` 工具 spawn 一个子 Agent，子 Agent 的 prompt 必须包含：

```
你是负责实现 [模块名] 的子 Agent。

## 任务清单
[从 tasks/<module-name>.md 中复制的完整 checklist]

## 详细设计
[从 detailed-design.md 中复制的该模块的 API 签名、数据结构、伪代码]

## 已完成的依赖模块代码
[列出该模块依赖的已完成文件的路径和关键接口摘要]

## 项目规范
- Python 3.10+，类型标注完整（mypy --strict）
- 代码风格符合 ruff 推荐规则集
- 所有公开函数/方法必须有 docstring
- 使用 numpy 进行数组操作
- 测试文件放在 tests/ 目录下，命名 test_<module>.py
- 测试必须可通过 `python -m pytest tests/test_<module>.py -v` 运行

## 交付物
1. 模块的 .py 源文件
2. 对应的 tests/test_<module>.py 测试文件
3. 测试执行结果（pytest 输出）

## 验证标准
- 所有 checklist 中的任务完成
- pytest 全部通过
- ruff check <file> 无错误
- mypy <file> 无错误
```

### 6.3 质量验证

子 Agent 完成后，你必须运行以下验证（**不可跳过**）：

```bash
cd /mnt/c/Users/35921/Desktop/本科学业/计算机博弈大赛
source venv/bin/activate

# 1. 单元测试
python -m pytest tests/test_<module>.py -v

# 2. 代码风格检查
ruff check src/<package>/<module>.py

# 3. 类型检查
mypy src/<package>/<module>.py
```

### 6.4 失败处理

如果任一步验证失败：

1. **读取失败输出**，分析错误原因
2. **自行修复代码**（用 Edit/Write 直接修改源文件或测试文件）
3. **重新运行验证**
4. 最多修复-验证循环 **3 次**
5. 3 次后仍失败 → 在 `tasks/progress.md` 中标记该模块为 ❌ 阻塞，输出详细失败报告（错误类型、已尝试的修复措施、建议方向），然后**暂停**

### 6.5 完成

验证全部通过后：
1. 在 `tasks/<module-name>.md` 中将所有完成的 checklist 子项从 `- [ ]` 改为 `- [x]`
2. 在 `tasks/progress.md` 中将该模块状态从 `☐ 未开始` 改为 `☑ 已完成`，填写完成时间
3. 输出简短完成报告：模块名、通过测试数、ruff/mypy 结果

---

## 7. 代码质量标准

### 7.1 类型标注

- 所有函数（公开和私有）**均须完整标注参数类型和返回值类型**
- 使用 `| None` 而非 `Optional[...]`（Python 3.10+）
- 使用 `list[tuple[int, int]]` 而非 `List[Tuple[int, int]]`
- 复杂类型可用 `typing.TypeAlias` 定义别名

### 7.2 命名规范（pep8-naming）

- 类名：`PascalCase`
- 函数/方法/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有方法/属性：`_leading_underscore`
- 模块级内部对象：`__all__` 列表显式声明公共接口

### 7.3 Docstring 规范

```python
def function(arg1: int, arg2: str) -> bool:
    """
    一句话描述函数功能。

    Args:
        arg1: 参数1的说明
        arg2: 参数2的说明

    Returns:
        返回值的说明

    Raises:
        ValueError: 何种情况下抛出
    """
```

### 7.4 测试规范

- 每个模块的测试文件至少覆盖：正常路径、边界条件、错误输入
- 测试函数命名：`test_<场景描述>`，如 `test_make_move_changes_player`
- 使用 `pytest.fixture` 避免重复的初始化代码
- 测试应独立——不依赖执行顺序

---

## 8. 关键设计决策速查

以下是详细设计文档中已确定的关键决策，子 Agent 必须遵循：

| 决策点 | 内容 |
|--------|------|
| 棋盘编码 | `EMPTY=0`, `WHITE=1`, `BLACK=-1`, `BURNED=2` |
| 坐标系统 | `(row, col)`，row 0-9（顶到底），col 0-9（左到右） |
| 走法格式 | `"a4-d4-g7"`（起点-移动终点-箭落点），代数记谱法 |
| 初始布局 | 黑方: d10, g10, a7, j7；白方: d1, g1, a4, j4（白方先行） |
| 白方值 | 1；黑方值：-1 |
| GUI | Tkinter（Python 标准库） |
| 配置 | YAML 文件，`config.yaml` |
| 策略头编码 | 三头分解（from/to/arrow 各 100 维）——第二阶段 |
| 置换表替换 | 深度优先（新 depth ≥ 旧 depth 时覆盖） |
| 错误处理 | 严格模式（非法走法抛异常） |

---

## 9. 必须遵守的原则

1. **不跳过验证**：每个模块完成后必须运行 pytest + ruff + mypy，不可省略
2. **不猜测接口**：子 Agent 实现时严格遵循 detailed-design.md 中的 API 签名
3. **不改需求**：所有实现以 detailed-design.md 为准，不擅自添加或删减功能
4. **模块隔离**：每个模块仅通过公开接口与其他模块交互，内部实现可自由调整
5. **失败透明**：遇到阻塞时清晰报告，不掩盖错误
6. **先跑通再优化**：第一阶段以功能正确为目标，性能优化可后续迭代

---

## 10. 启动指令

现在，请从**第一步：环境初始化**开始执行。完成环境初始化后，按**第 5 节**的顺序逐个完成模块。
