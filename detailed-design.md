# 亚马逊棋计算机博弈系统——详细设计文档

> **项目名称**：基于 Alpha-Beta 搜索与深度强化学习的亚马逊棋博弈系统  
> **文档版本**：v2.0  
> **日期**：2026-08-18  
> **主要开发者**：杨光  
> **依赖文档**：[需求文档 (proposal.md)](./proposal.md)

> **v2.0 变更说明**：本版在 v1.0（Alpha-Beta 方案）基础上，将第二阶段升级为**完全体 MCTS+ 深度网络方案**——覆盖 MCTS 搜索、策略-价值双头条件残差网络、自我对弈数据生成、训练流水线与「训练-评估-迭代」闭环，替换原来的启发式评估 + Alpha-Beta 作为主力引擎，Alpha-Beta 保留为兜底方案。

---

## 目录

1. [文档概述](#1-文档概述)
2. [模块总览与依赖关系](#2-模块总览与依赖关系)
3. [game 模块——游戏规则层](#3-game-模块游戏规则层)
4. [ai 模块——AI 搜索层](#4-ai-模块ai-搜索层)
5. [opening_book 模块——开局库](#5-opening_book-模块开局库)
6. [endgame 模块——残局库](#6-endgame-模块残局库)
7. [gui 模块——用户界面层](#7-gui-模块用户界面层)
8. [network 模块——深度学习层（MCTS+ 完整设计）](#8-network-模块深度学习层mcts-完整设计)
9. [main.py——程序入口](#9-mainpy程序入口)
10. [配置文件设计](#10-配置文件设计)
11. [测试计划](#11-测试计划)

---

## 1. 文档概述

### 1.1 文档目的

本文档对亚马逊棋博弈系统的**两阶段方案**进行详细设计，细化到每个模块的类/函数签名、数据结构、关键算法伪代码：

- **第一阶段（Alpha-Beta 方案，兜底）**：Minimax + Alpha-Beta 剪枝 + 启发式/蒙特卡洛评估，保证任意局面下都能给出合法走法。
- **第二阶段（完全体 MCTS+ 深度网络方案，主力）**：蒙特卡洛树搜索（MCTS）+ 策略-价值双头条件残差网络（§4.5、§8），配合自我对弈数据生成与训练闭环，替换启发式评估 + Alpha-Beta 作为主力引擎；Alpha-Beta 保留为训练未达标时的自动回退方案（见 §9）。

### 1.2 设计原则

- **模块独立**：每个模块仅通过公开接口与其他模块交互，内部实现可独立修改
- **可独立测试**：每个模块拥有对应的单元测试文件，测试不依赖其他模块的完整实现
- **先跑通再优化**：第一阶段优先保证功能完整与正确性，性能优化（置换表、走法排序）列为可插拔组件
- **接口先行**：所有跨模块交互均通过抽象基类或明确的函数签名定义

### 1.3 与需求文档的对应关系

本文档覆盖需求文档中的以下功能需求：

| 需求编号 | 功能 | 对应模块 |
|----------|------|----------|
| F-P0-01 | 棋盘状态表示 | `game/board.py` |
| F-P0-02 | 合法走法生成 | `game/rules.py` |
| F-P0-03 | 走法执行与撤销 | `game/board.py` |
| F-P0-04 | 终局判断 | `game/rules.py` |
| F-P0-05 | Alpha-Beta 搜索 | `ai/alphabeta.py` |
| F-P0-06 | 蒙特卡洛评估 | `ai/evaluator.py` |
| F-P0-07 | 迭代加深 | `ai/alphabeta.py` |
| F-P0-08 | 走法输出 | `game/move.py` |
| F-P0-09 | 走法输入 | `game/move.py` |
| F-P0-10 | GUI 显示 | `gui/` |
| F-P0-11 | 开局库 | `opening_book/` |
| F-P0-12 | 残局库 | `endgame/` |
| F-P2-01 | 置换表 | `ai/transposition.py` |
| F-P1-01 | 纯 MCTS 搜索（UCB1 基线） | `ai/mcts.py` |
| F-P1-02 | 策略-价值双头条件残差网络 | `network/model.py` |
| F-P1-03 | MCTS + 网络引导搜索 | `ai/mcts.py` + `network/model.py` |
| F-P1-04 | 自我对弈数据生成 | `network/selfplay.py` |
| F-P1-05 | 网络训练流水线 | `network/train.py` + `network/replay_buffer.py` |
| F-P1-06 | 训练-评估-迭代闭环 | `network/train.py` + `network/selfplay.py` |
| F-P1-07 | 模型持久化 | `network/train.py` |

---

## 2. 模块总览与依赖关系

### 2.1 模块分层架构

```
┌─────────────────────────────────────────────┐
│                  main.py                     │  ← 程序入口 + 训练入口
├─────────────────────────────────────────────┤
│              gui/ (用户界面)                  │  ← 可选层
├─────────────────────────────────────────────┤
│     ai/ (搜索引擎)            network/        │  ← 深度学习层
│     - alphabeta.py           - model.py      │
│     - mcts.py  ──────────►    - selfplay.py  │
│     - evaluator.py            - train.py     │
│     - transposition.py        - replay_      │
│       opening_book/             buffer.py    │
│       - book.py                - config.py   │
│       endgame/                               │
│       - tablebase.py                         │
├─────────────────────────────────────────────┤
│              game/ (游戏规则)                  │  ← 基础层
│     - board.py, rules.py, move.py,           │
│       constants.py                           │
└─────────────────────────────────────────────┘
```

### 2.2 模块依赖矩阵

```
                game/   ai/   opening_book/  endgame/  gui/  network/
game/            -      -         -             -        -      -
ai/              ✓      -         -             -        -   ✓(mcts)
opening_book/    ✓      -         -             -        -      -
endgame/         ✓      -         -             -        -      -
gui/             ✓      ✓         -             -        -      -
network/         ✓   ✓(selfplay)  -             -        -      -
main.py          ✓      ✓         ✓             ✓        ✓      -
```

> 箭头方向：A 依赖 B 表示 A 的代码中 import B。

> **依赖说明**：
> - `ai/mcts.py` 依赖 `network/model.py`（用网络做叶节点评估）——这是 ai 层对 network 层唯一的下行依赖。
> - `network/selfplay.py` 依赖 `ai/mcts.py`（用 MCTS 产生自我对弈走法）。两者不构成 Python 导入环：`mcts.py` 不反向导入 `selfplay.py`，仅在图示依赖方向上有往返，实际 import 链为 `selfplay → mcts → model`，单向无环。

### 2.3 模块独立性说明

| 模块 | 独立编译运行 | 独立测试方式 |
|------|-------------|-------------|
| `game/constants.py` | ✓（无依赖） | 直接验证常量值 |
| `game/move.py` | ✓（仅依赖 constants） | 测试坐标转换、字符串解析往返 |
| `game/board.py` | ✓（依赖 constants, move） | 用构造的走法列表测试 make/undo/clone |
| `game/rules.py` | ✓（依赖 board, constants） | 用构造的棋盘状态测试走法生成和终局判断 |
| `ai/evaluator.py` | ✓（依赖 game） | 用固定棋盘状态测试评估值范围 |
| `ai/alphabeta.py` | ✓（依赖 game, ai/evaluator, ai/transposition） | 用简单残局测试搜索是否返回合法走法 |
| `ai/mcts.py` | ✓（依赖 game, network/model） | 用固定棋盘 + 小模拟次数测试返回合法走法；一步必胜局面找到杀招 |
| `ai/transposition.py` | ✓（依赖 game） | 测试哈希一致性、存取往返 |
| `opening_book/book.py` | ✓（依赖 game） | 用预构造的 JSON 测试查询 |
| `endgame/tablebase.py` | ✓（依赖 game） | 用预计算的极小残局库测试查询 |
| `network/config.py` | ✓（无依赖） | 直接验证 dataclass 默认值 |
| `network/model.py` | ✓（依赖 game, network/config） | 测试 forward 输出形状、board_to_tensor 通道、价值域 |
| `network/replay_buffer.py` | ✓（无依赖） | 测试 push/sample/容量淘汰 |
| `network/selfplay.py` | ✓（依赖 game, ai/mcts, network/model） | 用小规模自对弈测试数据格式与 z 标注 |
| `network/train.py` | ✓（依赖 network/*） | 用随机数据测试损失可反向传播、checkpoint 往返 |
| `gui/` | ✓（依赖 game, ai） | 手动界面测试 |

---

## 3. game 模块——游戏规则层

### 3.1 constants.py

**职责**：定义全局常量，被所有其他模块引用。

#### 接口定义

```python
# 棋盘尺寸
BOARD_SIZE: int = 10  # 10×10 棋盘

# 格位编码
EMPTY: int = 0        # 空格
WHITE: int = 1        # 白方亚马逊
BLACK: int = -1       # 黑方亚马逊
BURNED: int = 2       # 烧毁格（障碍物）

# 棋盘坐标映射
# 代数记谱法: 列 a-j (col 0-9), 行 1-10 (row 9-0)
COL_LABELS: str = "abcdefghij"
ROW_LABELS: str = "12345678910"  # 注意 "10" 是两位数，解析时需特殊处理

# 初始布局
# 黑方 (row=0为第10行, row=9为第1行)
INITIAL_BLACK_POSITIONS: list[tuple[int, int]] = [
    (0, 3),   # d10
    (0, 6),   # g10
    (3, 0),   # a7
    (3, 9),   # j7
]
# 白方
INITIAL_WHITE_POSITIONS: list[tuple[int, int]] = [
    (6, 0),   # a4
    (6, 9),   # j4
    (9, 3),   # d1
    (9, 6),   # g1
]

# 皇后走法的 8 个方向 (row_delta, col_delta)
QUEEN_DIRECTIONS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# 胜负结果常量
WIN: int = 1
LOSE: int = -1
DRAW: int = 0  # 理论上亚马逊棋无平局，保留以备扩展
```

#### 坐标转换函数

```python
def pos_to_algebraic(row: int, col: int) -> str:
    """
    将 (row, col) 转换为代数记谱法字符串。
    例: (0, 3) → "d10", (9, 6) → "g1"
    
    注意: row 0 对应第 10 行, row 9 对应第 1 行
    col 0 对应 a 列, col 9 对应 j 列
    """

def algebraic_to_pos(notation: str) -> tuple[int, int]:
    """
    将代数记谱法字符串转换为 (row, col)。
    例: "d10" → (0, 3), "g1" → (9, 6)
    
    行号部分: "1"→row 9, "10"→row 0 (两位数字需特殊解析)
    
    Raises:
        ValueError: 记谱法格式非法
    """

def is_in_bounds(row: int, col: int) -> bool:
    """判断 (row, col) 是否在 10×10 棋盘范围内"""
```

### 3.2 move.py

**职责**：走法数据结构定义，走法字符串的编码与解码。

#### 数据结构

```python
@dataclass(frozen=True)
class Move:
    """不可变走法对象。frozen=True 使其可哈希，支持存入集合/字典。"""
    from_pos: tuple[int, int]   # 起点 (row, col)
    to_pos: tuple[int, int]     # 移动终点 (row, col)
    arrow_pos: tuple[int, int]  # 射箭落点 (row, col)
```

#### 接口定义

```python
class Move:
    # --- 属性 (由 @dataclass 自动生成) ---
    # from_pos: tuple[int, int]
    # to_pos: tuple[int, int]
    # arrow_pos: tuple[int, int]

    @staticmethod
    def from_string(s: str) -> Move:
        """
        从走法字符串构造 Move 对象。
        
        格式: "起点列行-终点列行-箭落点列行"
        例: "d10-d5-g7" 
            → Move(from_pos=(0,3), to_pos=(5,3), arrow_pos=(3,6))
        
        解析规则:
        1. 按 '-' 拆分为三段
        2. 每段调用 algebraic_to_pos() 解析
        3. 行号 "10" 需特殊处理（两位数字）
        
        Args:
            s: 走法字符串，格式为 "a1-b2-c3"
        
        Returns:
            Move 对象
        
        Raises:
            ValueError: 字符串格式非法或坐标越界
        """

    def to_string(self) -> str:
        """
        将 Move 转换为竞赛标准走法字符串。
        
        格式: "起点列行-终点列行-箭落点列行"
        例: Move(from_pos=(0,3), to_pos=(5,3), arrow_pos=(3,6))
            → "d10-d5-g7"
        
        Returns:
            标准走法字符串
        """

    def __repr__(self) -> str:
        """等价于 to_string()"""
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_move.py` | `from_string` ←→ `to_string` 往返一致性；非法字符串抛异常；坐标转换边界值 `"a1"`, `"j10"` |

### 3.3 board.py

**职责**：棋盘状态表示、走法执行/撤销、局面复制。这是整个系统中最核心的数据结构，被所有其他模块依赖。

#### 内部数据结构

```
Board 内部状态:

grid: np.ndarray, shape=(10,10), dtype=np.int8
    EMPTY(0) / WHITE(1) / BLACK(-1) / BURNED(2)

current_player: int
    1 (WHITE, 白方先行) 或 -1 (BLACK)

amazon_positions: dict[int, list[tuple[int, int]]]
    {WHITE: [(r,c), (r,c), (r,c), (r,c)], 
     BLACK: [(r,c), (r,c), (r,c), (r,c)]}
    用于快速获取己方/对方棋子位置，避免每次扫描整个棋盘。

zobrist_hash: int
    当前局面的 64 位 Zobrist 哈希值，随 make_move/undo_move 增量更新。

move_history: list[UndoInfo]
    走法历史栈，用于 undo_move()。每条记录包含:
    - from_pos, to_pos, arrow_pos: 原始走法
    - captured_from: 起点格的原值（通常为当前玩家棋子值）
    - captured_to: 移动终点格的原值（通常为 EMPTY）
    - captured_arrow: 箭落点格的原值（通常为 EMPTY）
    - prev_hash: 执行该走法前的 Zobrist 哈希值
    - prev_player: 执行该走法前的当前玩家
```

#### 接口定义

```python
class Board:
    def __init__(self):
        """
        初始化标准 10×10 棋盘。
        - 创建 10×10 全零矩阵
        - 按 INITIAL_BLACK_POSITIONS / INITIAL_WHITE_POSITIONS 放置棋子
        - current_player = WHITE (白方先行)
        - 初始化 amazon_positions 字典
        - 计算初始 Zobrist 哈希值
        - move_history 置空
        """

    # ========== 状态只读访问 ==========

    def get_grid(self) -> np.ndarray:
        """返回 (10,10) 棋盘的只读视图（或拷贝），供 rules.py 使用"""

    def get_piece_at(self, row: int, col: int) -> int:
        """返回 (row, col) 处的值: EMPTY/WHITE/BLACK/BURNED"""

    def get_current_player(self) -> int:
        """返回当前应走子的一方: WHITE(1) 或 BLACK(-1)"""

    def get_amazon_positions(self, player: int) -> list[tuple[int, int]]:
        """返回指定玩家所有亚马逊的位置列表"""

    def get_zobrist_hash(self) -> int:
        """返回当前局面的 64 位 Zobrist 哈希值"""

    # ========== 走法执行与撤销 ==========

    def make_move(self, move: Move) -> None:
        """
        在当前棋盘上执行一步走法，不创建新棋盘（原位修改以提升搜索性能）。
        
        执行步骤:
        1. 保存 UndoInfo 至 move_history 栈
        2. 将 from_pos 处的己方棋子移除（设为 EMPTY）
        3. 将 to_pos 处设为己方棋子
        4. 将 arrow_pos 处设为 BURNED
        5. 更新 amazon_positions（移除旧位置，加入新位置）
        6. 递增更新 zobrist_hash
        7. 切换 current_player
        
        Args:
            move: 待执行的走法
        
        Note:
            调用者需保证 move 为合法走法（由 rules.py 验证）。
            本方法不进行合法性检查以最大化搜索性能。
        """

    def undo_move(self) -> None:
        """
        撤销最近一步走法。
        从 move_history 栈顶弹出 UndoInfo，恢复所有被修改的状态。
        
        恢复步骤:
        1. 弹出栈顶 UndoInfo
        2. 恢复 from_pos, to_pos, arrow_pos 的原值
        3. 恢复 amazon_positions
        4. 恢复 zobrist_hash
        5. 恢复 current_player
        
        Raises:
            IndexError: move_history 为空时调用
        """

    def clone(self) -> 'Board':
        """
        深拷贝当前棋盘。
        用于蒙特卡洛评估中的随机模拟（不影响原始搜索状态）。
        
        Returns:
            与当前棋盘状态完全相同的新 Board 对象
        """

    # ========== 初始化 ==========

    def reset(self) -> None:
        """将棋盘重置到标准初始布局，清空走法历史"""
```

#### 关键算法：make_move 伪代码

```
make_move(board, move):
    # 1. 保存撤销信息
    undo = UndoInfo()
    undo.from_pos       = move.from_pos
    undo.to_pos         = move.to_pos
    undo.arrow_pos      = move.arrow_pos
    undo.captured_from  = grid[move.from_pos]
    undo.captured_to    = grid[move.to_pos]
    undo.captured_arrow = grid[move.arrow_pos]
    undo.prev_hash      = zobrist_hash
    undo.prev_player    = current_player
    
    # 2. 执行移动
    piece = grid[move.from_pos]          # WHITE 或 BLACK
    grid[move.from_pos]  = EMPTY
    grid[move.to_pos]    = piece
    grid[move.arrow_pos] = BURNED
    
    # 3. 更新棋子位置表
    amazon_positions[piece].remove(move.from_pos)
    amazon_positions[piece].append(move.to_pos)
    
    # 4. 增量更新 Zobrist 哈希
    zobrist_hash = zobrist_hash XOR hash_table['piece', move.from_pos, piece]
    zobrist_hash = zobrist_hash XOR hash_table['piece', move.to_pos, piece]
    zobrist_hash = zobrist_hash XOR hash_table['burned', move.arrow_pos]
    zobrist_hash = zobrist_hash XOR hash_table['side_to_move']
    
    # 5. 切换玩家
    current_player = -current_player
    
    # 6. 推入历史栈
    move_history.append(undo)
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_board.py` | 初始棋盘棋子位置正确性；make_move 后棋子位置、烧毁格、current_player 正确；undo_move 完全恢复；clone 独立性（修改 clone 不影响原对象）；Zobrist 哈希一致性（同一局面多次到达哈希相同） |

### 3.4 rules.py

**职责**：合法走法生成、终局判断、胜负判定。所有方法均为纯函数（无副作用），接受 Board 对象作为输入。

#### 接口定义

```python
class Rules:
    @staticmethod
    def get_legal_moves(board: Board) -> list[Move]:
        """
        生成当前局面下 current_player 的所有合法走法。
        
        走法生成流程（详见伪代码）:
        1. 遍历当前玩家的所有亚马逊位置
        2. 对每个亚马逊，生成其皇后走法可达的所有空格 → 移动候选集
        3. 对每个移动候选，从移动终点生成皇后走法可达的所有空格（排除棋子所在格）
           → 箭候选集
        4. 每个 (起点, 移动终点, 箭落点) 三元组为一个合法走法
        
        Args:
            board: 当前棋盘状态
        
        Returns:
            合法走法列表。若当前玩家无合法走法，返回空列表 []
        """

    @staticmethod
    def get_queen_reachable(board: Board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        返回从 from_pos 出发，皇后走法可达的所有空格位置。
        
        皇后走法: 沿 8 个方向 (QUEEN_DIRECTIONS) 任意步数，
        遇到棋盘边缘、棋子或烧毁格时停止该方向搜索。
        
        可达格的条件: grid[r][c] == EMPTY
        
        Args:
            board: 棋盘状态
            from_pos: 出发点 (row, col)
        
        Returns:
            可达空格位置列表
        """

    @staticmethod
    def is_game_over(board: Board) -> bool:
        """
        判断游戏是否结束。
        实现: 调用 get_legal_moves(board)，若返回空列表则游戏结束。
        
        Args:
            board: 棋盘状态
        
        Returns:
            True 表示当前玩家无合法走法，游戏结束
        """

    @staticmethod
    def get_winner(board: Board) -> int:
        """
        返回获胜方。
        
        Args:
            board: 棋盘状态（必须已终局）
        
        Returns:
            WHITE(1): 白方胜（黑方无合法走法）
            BLACK(-1): 黑方胜（白方无合法走法）
        
        Note:
            亚马逊棋不存在平局。应在 is_game_over() 返回 True 后调用。
        """
```

#### 关键算法：get_legal_moves 伪代码

```
get_legal_moves(board):
    moves = []
    player = board.get_current_player()
    amazons = board.get_amazon_positions(player)  # 当前玩家的 4 个亚马逊位置
    
    for amazon_pos in amazons:
        # 第一步：该亚马逊的合法移动目标
        move_targets = get_queen_reachable(board, amazon_pos)
        
        for move_target in move_targets:
            # 在临时棋盘上执行移动（仅修改棋子位置，不创建完整拷贝以提高性能）
            # 简化做法：直接在原棋盘上模拟
            temp_move_for_arrow = Move(amazon_pos, move_target, None)  # 暂不设箭位置
            board.make_move_partial(temp_move_for_arrow)  # 仅移动棋子
            
            # 第二步：从移动终点生成箭的合法落点
            arrow_targets = get_queen_reachable(board, move_target)
            
            # 第三步：恢复棋子位置
            board.undo_move_partial()
            
            # 生成完整走法
            for arrow_target in arrow_targets:
                moves.append(Move(amazon_pos, move_target, arrow_target))
    
    return moves
```

> **性能优化说明**：`make_move_partial` / `undo_move_partial` 是仅修改棋子位置（不烧格子）的轻量操作，用于避免在走法生成时为每个候选走法执行完整的 make/undo。如果这增加了实现复杂度，第一版可以直接用完整 make/undo，后续优化。

#### 关键算法：get_queen_reachable 伪代码

```
get_queen_reachable(board, from_pos):
    reachable = []
    (from_r, from_c) = from_pos
    
    for (dr, dc) in QUEEN_DIRECTIONS:       # 8 个方向
        r, c = from_r + dr, from_c + dc
        while is_in_bounds(r, c):
            if board.get_piece_at(r, c) == EMPTY:
                reachable.append((r, c))
                r += dr
                c += dc
            else:
                break   # 遇到棋子或烧毁格，此方向停止
    
    return reachable
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_rules.py` | 初始棋盘合法走法数量正确（手工计算参考值）；已知终局局面 is_game_over() 返回 True；get_queen_reachable 在空旷棋盘上可达格数正确（中心位置应可达 27 格）；有障碍物时 correctly blocked |

---

## 4. ai 模块——AI 搜索层

### 4.1 engine.py

**职责**：定义所有 AI 搜索引擎的抽象基类，统一搜索接口。

#### 接口定义

```python
from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def search(self, board: 'Board', time_budget_ms: float) -> 'Move':
        """
        在给定棋盘状态和时间预算内，搜索并返回最优走法。
        
        Args:
            board: 当前棋盘状态（不会被修改，Engine 内部会 clone）
            time_budget_ms: 可用时间预算（毫秒）。若为 0 或负数，表示无限制。
        
        Returns:
            搜索到的最优走法
        
        Note:
            即使时间耗尽，也必须返回一个合法走法（如第一个合法走法作为保底）。
        """

    @abstractmethod
    def get_name(self) -> str:
        """返回引擎名称，用于日志和评估输出。"""

    @abstractmethod
    def reset(self) -> None:
        """重置引擎内部状态（置换表、杀手走法等），用于新一局开始时调用。"""
```

---

### 4.2 evaluator.py

**职责**：局面评估函数。第一阶段实现蒙特卡洛随机模拟评估（需求 F-P0-06）。

#### 接口定义

```python
class MonteCarloEvaluator:
    def __init__(self, num_simulations: int = 100):
        """
        Args:
            num_simulations: 每次评估执行的随机模拟局数，默认 100
        """

    def evaluate(self, board: Board) -> float:
        """
        从当前局面出发，执行 num_simulations 次随机对弈，
        返回当前玩家的估计胜率。
        
        Args:
            board: 待评估的局面（不会被修改）
        
        Returns:
            float, 范围 [0.0, 1.0]
            1.0 = 当前玩家必胜, 0.0 = 当前玩家必败
        """

    def set_num_simulations(self, n: int) -> None:
        """动态调整模拟次数"""
```

#### 关键算法：evaluate 伪代码

```
evaluate(board):
    player = board.get_current_player()
    wins = 0
    
    for i in 1..num_simulations:
        sim_board = board.clone()         # 深拷贝，避免影响原始状态
        result = random_playout(sim_board)
        if result == player:              # 当前玩家胜
            wins += 1
        # else: 对方胜（amazons 无平局）
    
    return wins / num_simulations


random_playout(board):
    """
    从当前局面出发，双方以均匀随机策略对弈至终局。
    返回胜者: WHITE 或 BLACK。
    """
    while not Rules.is_game_over(board):
        moves = Rules.get_legal_moves(board)
        if not moves:
            break
        move = random.choice(moves)        # 均匀随机选择
        board.make_move(move)
    
    return Rules.get_winner(board)
```

#### 性能考量

- 每次 `evaluate` 执行 `num_simulations × (平均每局步数 × 走法生成成本)` 
- 在 Alpha-Beta 的叶节点调用，调用频次 = 搜索树叶子数
- **关键优化**：`num_simulations` 需要根据搜索深度动态调整——深度较大时（如 layer 4-5），可减少模拟次数（如 30-50 次）以减少叶节点评估时间

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_evaluator.py` | 固定种子下多次评估返回相同值；评估值始终在 [0,1] 范围；必胜局面（对手无合法走法）返回 1.0；模拟次数变化时值合理变化 |

---

### 4.3 alphabeta.py

**职责**：Alpha-Beta 剪枝搜索 + 迭代加深（需求 F-P0-05, F-P0-07）。

#### 内部数据结构

```python
@dataclass
class SearchResult:
    """一次搜索的返回值"""
    move: Move              # 最优走法
    value: float            # 该走法的评估值 [0, 1]

@dataclass
class SearchStats:
    """搜索统计信息，用于性能分析和调优"""
    nodes_visited: int
    leaves_evaluated: int
    transposition_hits: int
    depth_reached: int
    time_elapsed_ms: float
```

#### 接口定义

```python
class AlphaBetaEngine(Engine):
    def __init__(
        self,
        evaluator: MonteCarloEvaluator,
        max_depth: int = 5,
        time_budget_ms: float = 15000.0,          # 单步最大时间（比赛默认）
        use_transposition_table: bool = True,
        use_iterative_deepening: bool = True,
        use_move_ordering: bool = False,           # P2 优化，初版默认关闭
        transposition_table_size: int = 2**20,     # 约 100 万条目
    ):
        """
        Args:
            evaluator: 局面评估器
            max_depth: 最大搜索深度（迭代加深的最终目标深度）
            time_budget_ms: 单步时间预算，默认 15 秒
            use_transposition_table: 是否启用置换表
            use_iterative_deepening: 是否启用迭代加深
            use_move_ordering: 是否启用法排序（第一版关）
            transposition_table_size: 置换表容量
        """

    # ========== Engine 接口实现 ==========

    def search(self, board: Board, time_budget_ms: float) -> Move:
        """
        搜索入口。根据配置选择迭代加深或固定深度搜索。
        
        流程:
        1. 记录开始时间
        2. 若启用开局库，先查询开局库
        3. 若启用迭代加深 → iterative_deepening_search()
        4. 否则 → fixed_depth_search()
        5. 返回找到的最优走法
        """

    def get_name(self) -> str:
        return f"AlphaBeta(depth={self.max_depth})"

    def reset(self) -> None:
        """清空置换表和内部统计"""

    # ========== 搜索核心 ==========

    def iterative_deepening_search(
        self, board: Board, start_time: float, time_budget_ms: float
    ) -> SearchResult:
        """
        迭代加深搜索。
        
        伪代码见下文。
        """

    def fixed_depth_search(
        self, board: Board, depth: int
    ) -> SearchResult:
        """
        固定深度搜索（无时间限制），用于测试和评估。
        """

    def _alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        start_time: float,
        time_budget_ms: float,
    ) -> float:
        """
        Alpha-Beta 递归搜索。
        
        Args:
            board: 当前局面（会被 make/undo 修改，但退出时恢复）
            depth: 剩余搜索深度（0 = 叶节点）
            alpha: 当前最佳下界
            beta: 当前最佳上界
            is_maximizing: True 表示当前层为极大层
            start_time: 搜索开始时间戳
            time_budget_ms: 时间预算
        
        Returns:
            该局面的评估值 [0, 1]
        
        伪代码见下文。
        """

    # ========== 辅助方法 ==========

    def _check_timeout(self, start_time: float, time_budget_ms: float) -> bool:
        """检查是否超时（当前时间 - start_time > time_budget_ms * 0.8）"""

    def _get_move_priority(self, move: Move) -> int:
        """
        走法排序启发式（初版可返回固定值）。
        后续可基于历史表、杀手走法进行排序。
        """

    def get_stats(self) -> SearchStats:
        """返回最近一次搜索的统计信息"""

    def set_opening_book(self, book: 'OpeningBook') -> None:
        """注入开局库（可选依赖）"""

    def set_endgame_tablebase(self, tablebase: 'Tablebase') -> None:
        """注入残局库（可选依赖）"""
```

#### 关键算法：iterative_deepening_search 伪代码

```
iterative_deepening_search(board, start_time, time_budget_ms):
    best_result = SearchResult(
        move = Rules.get_legal_moves(board)[0],  # 保底: 第一个合法走法
        value = 0.0
    )
    
    for depth = 1 to max_depth:
        if _check_timeout(start_time, time_budget_ms):
            break
        
        # 执行固定深度搜索
        result = fixed_depth_search(board, depth, start_time, time_budget_ms)
        
        if not timeout_occurred:
            best_result = result
        
        # 若本层搜索完成且 value 接近 1.0（必胜），可提前终止
        if best_result.value > 0.99:
            break
    
    stats.depth_reached = depth - 1
    return best_result


fixed_depth_search(board, depth, start_time, time_budget_ms):
    moves = Rules.get_legal_moves(board)
    if not moves:
        return SearchResult(None, 0.0)  # 无合法走法
    
    best_move = moves[0]
    best_value = 0.0
    
    for move in moves:
        board.make_move(move)
        value = _alpha_beta(board, depth-1, 0.0, 1.0, False, start_time, time_budget_ms)
        board.undo_move()
        
        if _check_timeout(start_time, time_budget_ms):
            break
        
        if value > best_value:
            best_value = value
            best_move = move
    
    return SearchResult(best_move, best_value)
```

#### 关键算法：_alpha_beta 伪代码

```
_alpha_beta(board, depth, alpha, beta, is_maximizing, start_time, time_budget_ms):
    # --- 终止条件 ---
    
    # 条件1: 超时
    if _check_timeout(start_time, time_budget_ms):
        raise TimeoutError  # 向上传播，由 iterative_deepening_search 捕获
    
    # 条件2: 终局
    if Rules.is_game_over(board):
        winner = Rules.get_winner(board)
        # 返回当前玩家视角的胜负: 1.0(己胜) / 0.0(己负)
        current = board.get_current_player()
        return 1.0 if winner == current else 0.0
    
    # 条件3: 搜索深度耗尽 → 评估
    if depth == 0:
        stats.leaves_evaluated += 1
        
        # 查询置换表
        if use_transposition_table:
            tt_entry = transposition_table.lookup(board.get_zobrist_hash())
            if tt_entry and tt_entry.depth >= 0:
                stats.transposition_hits += 1
                return tt_entry.value
        
        return evaluator.evaluate(board)
    
    # --- 走法生成与排序 ---
    moves = Rules.get_legal_moves(board)
    if not moves:
        return 0.0  # 当前玩家负
    
    if use_move_ordering:
        moves = sort_moves_by_priority(moves)
    
    # --- Alpha-Beta 搜索 ---
    stats.nodes_visited += 1
    
    if is_maximizing:
        value = -inf
        for move in moves:
            board.make_move(move)
            value = max(value, _alpha_beta(board, depth-1, alpha, beta, False, ...))
            board.undo_move()
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta 剪枝
        return value
    else:
        value = +inf
        for move in moves:
            board.make_move(move)
            value = min(value, _alpha_beta(board, depth-1, alpha, beta, True, ...))
            board.undo_move()
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha 剪枝
        return value
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_alphabeta.py` | depth=1 搜索找到直接获胜走法（构造必胜一步杀局面）；depth=2 不选导致己方立即输的走法；迭代加深在超时后能返回合法走法；搜索结果与穷举一致（极小残局） |
| `tests/test_alphabeta_integration.py` | 完整对弈: AlphaBeta vs Random, 100 局胜率 > 90% |

---

### 4.4 transposition.py

**职责**：Zobrist 哈希表与置换表（需求 F-P2-01）。为可插拔优化组件，AlphaBetaEngine 可选择启用。

#### Zobrist 哈希设计

```
哈希键组成:
- 对每个 (piece_type, row, col) 组合生成 64 位随机数
  piece_type ∈ {WHITE, BLACK, BURNED}
  共 3 × 10 × 10 = 300 个随机数
- 一个额外的 64 位随机数表示"当前走子方"
  共 1 个随机数

哈希更新 (增量):
  hash = hash XOR table[old_piece, row, col]   // 移除旧值
  hash = hash XOR table[new_piece, row, col]   // 置入新值

初始哈希:
  hash = side_to_move_rand
  for each initial piece:
      hash = hash XOR table[piece_type, row, col]
```

#### 接口定义

```python
class ZobristHasher:
    """Zobrist 哈希生成器。随机数在首次实例化时生成并缓存（类级别）。"""

    _table: dict[tuple[int, int, int], int] = {}  # (piece, row, col) → 64-bit random
    _side_to_move_rand: int = 0
    _initialized: bool = False

    @classmethod
    def init(cls) -> None:
        """使用固定种子生成所有随机数以保证可复现性"""

    @classmethod
    def hash_piece(cls, piece: int, row: int, col: int) -> int:
        """返回 (piece, row, col) 对应的随机哈希值"""

    @classmethod
    def hash_side_to_move(cls) -> int:
        """返回当前走子方的哈希分量"""


class TranspositionEntry:
    """置换表条目"""
    hash_key: int      # 64 位哈希（用于冲突检测）
    depth: int         # 该条目记录的搜索深度
    value: float       # 评估值 [0, 1]
    flag: int          # EXACT / LOWER_BOUND / UPPER_BOUND
    best_move: Move    # 该局面下的最佳走法（用于走法排序）


class TranspositionTable:
    """置换表。使用简单数组 + 取模实现，发生冲突时深度优先替换。"""

    EXACT: int = 0
    LOWER_BOUND: int = 1
    UPPER_BOUND: int = 2

    def __init__(self, size: int = 2**20):
        """
        初始化指定容量的置换表。
        size 应为 2 的幂（便于位运算取模）。
        """

    def lookup(self, hash_key: int) -> TranspositionEntry | None:
        """查询哈希值对应的条目。若不存在或哈希不匹配返回 None。"""

    def store(self, hash_key: int, depth: int, value: float, 
              flag: int, best_move: Move = None) -> None:
        """
        存入置换表。
        替换策略: 若槽位为空或新 depth ≥ 旧 depth，则覆盖。
        """

    def clear(self) -> None:
        """清空置换表"""
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_transposition.py` | 同一局面多次到达哈希一致；不同局面哈希碰撞率低；store → lookup 往返得到相同值；clear 后 lookup 返回 None |

---

### 4.5 mcts.py

**职责**：蒙特卡洛树搜索。实现**网络引导的 MCTS 搜索引擎**（需求 F-P1-03）作为第二阶段主力引擎，并提供**无网络的纯 MCTS 基线**（需求 F-P1-01）用于棋力对照。两者均实现 `Engine` 接口，与 `AlphaBetaEngine` 可互换（§9）。

#### 内部数据结构

```python
@dataclass
class MCTSNode:
    """MCTS 搜索树节点。"""

    board: Board                      # 该节点对应的棋盘快照（clone）
    move: Move | None                 # 到达此节点的走法（根节点为 None）
    parent: MCTSNode | None           # 父节点
    children: dict[int, MCTSNode]     # 已展开子节点：move 键 → 子节点
    untried_moves: list[Move]         # 尚未展开的合法走法（叶节点非空）
    priors: dict[int, float]          # move 键 → 先验概率 P(s, a)
    n_visits: int = 0                 # 访问次数 N
    total_value: float = 0.0          # 累计价值 W（始终以本节点行棋方视角）
    is_terminal: bool = False         # 是否终局
    terminal_value: float = 0.0       # 终局价值（本节点视角，见下方说明）

    def q_value(self) -> float:
        """平均价值 Q = W / N；N == 0 时返回 0.0。"""
```

> **move 键**：`Move` 为 `frozen=True` 的 dataclass，可直接哈希，用 `Move` 对象本身（或其 `(from_pos, to_pos, arrow_pos)` 三元组）作为 `children`/`priors` 的字典键。

> **视角约定**：`total_value` 与 `terminal_value` 始终以**本节点当前行棋方**的视角记录。终局节点的行棋方无合法走法 → 判负，故 `terminal_value = -1.0`。回传时每上溯一层取相反数完成视角翻转（与 §4.3 Alpha-Beta 的 `1.0 - value` 等价，但值域为 `[-1, +1]`）。

#### 接口定义

```python
class MCTSEngine(Engine):
    """网络引导的 MCTS 搜索引擎（AlphaZero 风格）。"""

    def __init__(
        self,
        model: AmazonsNet,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 0.0,        # 正式对弈用 0（贪心取访问最多者）
        dirichlet_alpha: float = 0.3,    # 仅 self-play 根节点注入噪声
        dirichlet_epsilon: float = 0.25,
    ): ...

    def search(self, board: Board, time_budget_ms: float) -> Move:
        """执行 num_simulations 次模拟，返回访问次数最多的走法（temperature≈0）。

        Args:
            board: 当前局面（不会被修改，内部 clone）
            time_budget_ms: 时间预算。MCTS 以固定模拟次数为准，此参数仅用于
                兼容 Engine 接口——若剩余时间不足则提前截断模拟。

        Returns:
            搜索到的最优走法（根节点子节点中 N 最大者）。
        """

    def get_name(self) -> str:
        return f"MCTS(simulations={self._num_simulations})"

    def reset(self) -> None:
        """清空统计。MCTS 每步重建搜索树，reset 主要重置累计统计。"""

    # ---------- 供 self-play 复用的接口 ----------

    def get_policy(self, board: Board, temperature: float) -> tuple[dict[int, float], Move]:
        """
        搜索后返回 (走法访问次数分布 π, 采样走法)。

        π 以 {move 键: 概率} 形式返回，供 selfplay 记录训练标签；
        采样走法按温度 τ 从 π 中采样（τ=0 贪心）。
        """
```

#### 关键算法：搜索主循环

```
search(board, time_budget_ms):
    root = MCTSNode(board=clone(board), move=None, parent=None)
    root.untried_moves = Rules.get_legal_moves(root.board)
    if root.untried_moves 为空:
        raise ValueError("search() called on terminal position")

    for sim in 1..num_simulations:
        if 超时(time_budget_ms):
            break

        node = root
        path = [root]

        # 1. 选择（Select）：沿树下降到叶节点
        while node.untried_moves 为空 and not node.is_terminal:
            node = node.select_child(c_puct)
            path.append(node)

        # 2. 扩展 + 评估（Expand & Evaluate）
        if node.is_terminal:
            value = node.terminal_value          # -1.0（本节点视角）
        else:
            value = node.expand(model)           # 网络评估 → 先验 + 价值 v

        # 3. 回传（Backpropagate）：每上溯一层视角翻转
        for n in reversed(path):
            n.n_visits += 1
            n.total_value += value
            value = -value                       # 父节点为对方视角

    # 4. 走法选择
    return root.select_move(temperature=0.0)
```

#### 关键算法：PUCT 选择

```
select_child(c_puct):
    # 对每个已展开子节点 child：
    #   Q = child.total_value / child.n_visits       (n_visits > 0)
    #   U = c_puct * child.priors[child.move]
    #       * sqrt(self.n_visits) / (1 + child.n_visits)
    # 返回 argmax( Q + U )
```

> **首次访问**：子节点刚创建时 `n_visits = 0`，`Q` 取 0.0，仅由先验 `P` 与探索项驱动，保证所有合法走法最终都会被尝试。

#### 关键算法：扩展（网络评估 + 建子节点）

```
expand(node, model):
    # 1. 网络前向：返回条件三头策略 logits 与价值 v ∈ [-1, 1]（node 当前行棋方视角）
    logits, v = model.predict(node.board)

    # 2. 计算每个合法走法的先验 P(move)（条件三头分解，见 §8.2.3）
    for move in node.untried_moves:
        priors[move] = P(from) · P(to | from) · P(arrow | to)

    # 3. 数值归一化（理论上已归一，防止浮点漂移）
    priors = normalize(priors)

    # 4. 为每个合法走法创建子节点
    for move in node.untried_moves:
        child_board = node.board.clone()
        child_board.make_move(move)
        child = MCTSNode(board=child_board, move=move, parent=node,
                         prior=priors[move])
        child.untried_moves = Rules.get_legal_moves(child_board)
        if child.untried_moves 为空:
            child.is_terminal = True
            child.terminal_value = -1.0      # 对方无走法 → 本节点视角对方负
        node.children[key(move)] = child

    node.untried_moves = []
    return v
```

> **性能说明**：亚马逊棋分支因子可达 2000+，展开一次即需 `clone + make_move` 约 2000 次。第一版采用上述**急切展开**（正确性优先）；后续可优化为**懒展开**——`select_child` 首次进入某子节点时才 `clone` 并生成其 `untried_moves`，避免一次网络评估即克隆全部子棋盘。

#### 关键算法：走法选择（温度 τ）

```
select_move(temperature):
    visits = [child.n_visits for child in children]
    if temperature == 0:
        return argmax(visits) 对应的 move
    else:
        probs = visits^(1/temperature)
        probs = probs / sum(probs)
        return sample(probs)
```

#### 纯 MCTS 基线（需求 F-P1-01）

```python
class PureMCTSEngine(Engine):
    """无网络引导的 MCTS 基线：UCB1 选择 + 随机模拟（rollout）评估。

    与 MCTSEngine 复用同一套 MCTSNode 树结构，差异仅在于：
    - 选择用 UCB1：Q + c * sqrt(ln N_parent / (1 + N_child))，无先验项。
    - 叶节点评估：从该节点出发双方均匀随机走子至终局（复用
      evaluator.py 的随机模拟逻辑），返回本节点视角的 +1/-1。
    - 先验 P 恒为均匀分布（或省略先验项）。
    """

    def __init__(self, num_simulations: int = 2000, c_puct: float = 1.0): ...
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_mcts.py` | 返回走法合法；终局局面抛 ValueError 或返回 None；构造一步必胜局面搜索能找到杀招；访问次数随模拟次数增加而增长；`PureMCTSEngine` 同样返回合法走法 |

---

## 5. opening_book 模块——开局库

### 5.1 book.py

**职责**：加载开局库数据，支持哈希查询与加权随机走法选择（需求 F-P0-11）。

#### 数据文件格式 (`opening_book.json`)

```json
{
    "version": "1.0",
    "description": "Amazons opening book - generated from self-play",
    "positions": {
        "a1b2c3d4e5f6": [
            {
                "move": "d1-d5-g8",
                "weight": 100,
                "comment": "主变: 中心控制"
            },
            {
                "move": "g1-d4-d8",
                "weight": 80,
                "comment": "副变: 侧翼展开"
            }
        ],
        "f6e5d4c3b2a1": [
            {
                "move": "d10-d6-g3",
                "weight": 100
            }
        ]
    }
}
```

> 键名为局面的 Zobrist 哈希值的十六进制字符串表示。

#### 接口定义

```python
class OpeningBook:
    def __init__(self, book_path: str = None):
        """
        加载开局库 JSON 文件。
        
        Args:
            book_path: JSON 文件路径，默认为 None（禁用开局库）
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: JSON 格式错误
        """

    def query(self, board_hash: int) -> list[dict] | None:
        """
        查询指定局面的开局走法。
        
        Args:
            board_hash: 64 位 Zobrist 哈希值
        
        Returns:
            list of {"move": str, "weight": int, "comment": str} 若命中
            None 若未命中
        """

    def get_random_move(self, board_hash: int) -> Move | None:
        """
        加权随机选取一个开局走法。
        
        算法:
        1. query(board_hash) 获取候选列表
        2. 若未命中返回 None
        3. 以 weight 为权重做加权随机采样
        4. 用 weight/sum(weights) 作为概率分布
        5. 将选中的走法字符串转换为 Move 对象
        
        Args:
            board_hash: 64 位 Zobrist 哈希值
        
        Returns:
            选中的 Move 对象，或 None（未命中）
        """

    def is_available(self) -> bool:
        """开局库是否已加载"""

    def get_total_positions(self) -> int:
        """返回开局库中的局面总数"""
```

#### 集成方式

在 `AlphaBetaEngine.search()` 中：

```python
def search(self, board, time_budget_ms):
    if self.opening_book and self.opening_book.is_available():
        book_move = self.opening_book.get_random_move(board.get_zobrist_hash())
        if book_move is not None:
            return book_move  # 开局库命中，直接返回，不搜索
    
    # 否则正常搜索
    return self.iterative_deepening_search(...)
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_opening_book.py` | 命中情况返回非 None；未命中返回 None；加权随机分布统计检验；JSON 格式错误的异常处理 |

---

## 6. endgame 模块——残局库

### 6.1 tablebase.py

**职责**：残局库查询接口，从预计算的 `.pkl` 文件加载数据（需求 F-P0-12）。

#### 接口定义

```python
class EndgameResult:
    """残局查询结果"""
    result: int    # WIN(1) / LOSE(-1)
    distance: int  # 距终局的步数


class Tablebase:
    def __init__(self, data_path: str = None):
        """
        加载预计算的残局库。
        
        Args:
            data_path: .pkl 文件路径，默认为 None（禁用）
        """

    def query(self, board_hash: int) -> EndgameResult | None:
        """
        查询残局库。
        
        Args:
            board_hash: 64 位 Zobrist 哈希值
        
        Returns:
            EndgameResult 若命中，None 若未命中
        """

    def is_available(self) -> bool:
        """残局库是否已加载并可用"""
```

#### 集成方式

在 `AlphaBetaEngine._alpha_beta()` 中，于蒙特卡洛评估之前先查询残局库：

```python
def _alpha_beta(self, board, depth, ...):
    if depth == 0 or Rules.is_game_over(board):
        # 先查残局库
        if self.tablebase and self.tablebase.is_available():
            tb_result = self.tablebase.query(board.get_zobrist_hash())
            if tb_result is not None:
                current = board.get_current_player()
                if tb_result.result == WIN:
                    return 1.0  # (简化处理，忽略 distance)
                else:
                    return 0.0
        
        return self.evaluator.evaluate(board)
    ...
```

### 6.2 retrograde.py

**职责**：残局库预计算——回溯分析生成器（初赛后执行，在线/离线均可）。

#### 预计算范围

| 条件 | 值 |
|------|-----|
| 双方剩余棋子总数 | 2-3 子 |
| 可走空格数 | ≤ 15 格 |
| 棋盘上总自由格 | ≤ 15（含棋子和空格，不含烧毁格） |

#### 接口定义

```python
class RetrogradeGenerator:
    def __init__(self, max_pieces: int = 3, max_empty: int = 15):
        """
        Args:
            max_pieces: 双方棋子总数上限
            max_empty: 棋盘上总空闲格数上限（含空格，不含烧毁格）
        """

    def generate(self, output_path: str) -> dict:
        """
        执行回溯分析，生成残局库。
        
        算法概要:
        1. 枚举所有满足条件的子局面（棋子位置组合 × 空格/烧毁格排列）
        2. 标记所有终局状态（某方无合法走法）
        3. 反向迭代:
           - 若从 S 可走到已标记为 LOSE 的局面，则 S 为 WIN
           - 若从 S 所有走法都走向已标记为 WIN 的局面，则 S 为 LOSE
        4. 迭代至不动点（无新标记产生）
        5. 序列化保存，记录进度日志
        
        Returns:
            dict: {hash_hex: {"result": "WIN"/"LOSE", "distance": N}, ...}
        """

    def estimate_state_space(self) -> int:
        """估算状态空间大小，用于判断计算是否可行"""
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_endgame.py` | 极小残局库（如 2 子 4 空格）回溯分析结果正确；已知必胜/必败局面查询正确 |

---

## 7. gui 模块——用户界面层

**技术选型**：Tkinter（Python 标准库，零依赖安装）。

### 7.1 整体架构

```
gui/
├── main_window.py    → AmazonsGUI(tk.Tk)      主窗口控制器
├── board_painter.py  → BoardCanvas(tk.Canvas) 棋盘绘制组件
└── dialogs.py        → 对话框工具函数
```

### 7.2 main_window.py

#### 接口定义

```python
class AmazonsGUI:
    """
    主 GUI 窗口。
    
    运行模式:
    - HUMAN_VS_AI: 人类执白，AI 执黑
    - AI_VS_HUMAN: AI 执白，人类执黑
    - AI_VS_AI:   观战模式，双方均为 AI
    """

    def __init__(self, engine: Engine, mode: str = "HUMAN_VS_AI"):
        """
        Args:
            engine: AI 搜索引擎实例
            mode: 游戏模式
        """

    def run(self) -> None:
        """启动 GUI 主循环 (tk.mainloop)"""

    # --- 内部方法 ---
    def _on_board_click(self, row: int, col: int) -> None:
        """
        棋盘点击事件处理——三阶段状态机:
        
        State IDLE (未选择棋子):
            click → 若点击的是己方亚马逊 → State AMAZON_SELECTED
                  → 否则忽略
        
        State AMAZON_SELECTED (已选棋子，等待移动目标):
            click → 若是合法移动目标 → State MOVE_SELECTED
                  → 若是己方另一亚马逊 → 切换选择
                  → 否则忽略
        
        State MOVE_SELECTED (已选移动目标，等待箭落点):
            click → 若是合法箭落点 → 执行走法 → State IDLE
                  → 否则忽略
        """

    def _on_ai_move(self) -> None:
        """触发 AI 走子。在新线程中执行搜索以避免阻塞 GUI。"""

    def _update_display(self) -> None:
        """同步棋盘显示与内部状态"""

    def _on_new_game(self) -> None:
        """新局按钮回调"""

    def _on_undo(self) -> None:
        """悔棋按钮回调（人机模式中撤销人类+AI各一步）"""
```

#### 人类走法输入状态机

```
           click己方亚马逊
IDLE ─────────────────────────→ AMAZON_SELECTED
                                      │
          click己方另一亚马逊           │ click合法移动目标
          (切换选择)                    │
            ┌──────────────────────────┘
            ▼
      AMAZON_SELECTED ─────────→ MOVE_SELECTED
                                      │
                              click合法箭落点
                                      │
                                      ▼
                                    IDLE
                              (走法传给 Board.make_move)
```

### 7.3 board_painter.py

#### 接口定义

```python
class BoardCanvas(tk.Canvas):
    """棋盘绘制画布，嵌入主窗口"""

    # 视觉常量
    CELL_SIZE: int = 60          # 每格像素大小
    PIECE_RADIUS: int = 22       # 棋子半径
    ARROW_RADIUS: int = 5        # 箭孔标记半径
    BOARD_PADDING: int = 30      # 棋盘边距
    BOARD_COLOR: str = "#DEB887" # 棋盘底色 (木色)
    LINE_COLOR: str = "#000000"  # 网格线颜色

    def __init__(self, parent):
        """
        初始化画布尺寸: (CELL_SIZE * 10 + 2 * PADDING) × 同
        
        绑定鼠标点击事件 → 转换为棋盘坐标后回调
        """

    def set_click_callback(self, callback: callable) -> None:
        """注册棋盘点击回调: callback(row, col)"""

    def draw_board(self, board: Board) -> None:
        """
        绘制完整棋盘:
        1. 清空画布
        2. 绘制 10×10 网格线
        3. 绘制列标 (a-j) 顶部 + 行标 (1-10) 右侧
        4. 遍历 grid，绘制棋子 / 烧毁格
        5. 高亮最近一步走法的移动路径（箭头线）
        """

    def highlight_last_move(self, move: Move) -> None:
        """
        高亮显示最新走法:
        - from_pos → to_pos: 绘制箭头线 + 半透明色块
        - arrow_pos: 特殊标记 (红色 × 或爆炸图标)
        """

    def _pixel_to_board(self, x: int, y: int) -> tuple[int, int]:
        """像素坐标 → 棋盘行列坐标 (row, col)，点击在格子外返回 None"""

    def _draw_piece(self, row: int, col: int, player: int) -> None:
        """
        绘制一枚棋子:
        - WHITE: 白色圆形 + 黑色边框
        - BLACK: 黑色圆形 + 白色边框
        """

    def _draw_burned(self, row: int, col: int) -> None:
        """绘制烧毁格: 灰色方块 + × 标记"""
```

### 7.4 dialogs.py

```python
def show_game_over_dialog(parent, winner: int) -> None:
    """游戏结束弹窗，显示胜者并询问是否新局"""

def show_new_game_dialog(parent) -> str | None:
    """
    新局设置弹窗，选择模式:
    - 人类执白 vs AI
    - AI vs 人类执黑
    - AI vs AI (观战)
    """

def show_error_dialog(parent, message: str) -> None:
    """错误提示弹窗（如非法走法输入）"""
```

#### 独立测试说明

GUI 模块测试以手工界面测试为主，不要求自动化测试。至少需验证：
- 初始棋盘渲染正确
- 三阶段走法输入状态机正确流转
- 非法点击被正确忽略
- AI vs AI 模式棋盘正确更新

---

## 8. network 模块——深度学习层（MCTS+ 完整设计）

> **说明**：本章为第二阶段完整设计，替换 v1.0 的接口占位。实现 MCTS 搜索所需的策略-价值双头**条件**残差网络、自我对弈数据生成、经验回放池、训练流水线与「训练-评估-迭代」闭环。

### 8.1 config.py

```python
@dataclass
class NetworkConfig:
    """网络架构超参数。"""
    board_size: int = 10
    input_channels: int = 6            # 5 核心通道 + 1 可选「最近一步标记」通道
    num_residual_blocks: int = 13
    num_filters: int = 256
    policy_from_dim: int = 100         # 起点 10×10
    policy_to_dim: int = 100           # 移动终点 10×10
    policy_arrow_dim: int = 100        # 箭落点 10×10
    value_hidden_dim: int = 256


@dataclass
class TrainingConfig:
    """训练超参数。"""
    batch_size: int = 512
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"          # 余弦退火
    training_steps_per_cycle: int = 1000
    replay_buffer_capacity: int = 300000 # 最大样本数（FIFO 淘汰）
    self_play_games_per_cycle: int = 200
    eval_games: int = 100
    eval_win_rate_threshold: float = 0.55
    num_training_cycles: int = 100


@dataclass
class MCTSConfig:
    """MCTS 搜索超参数。"""
    num_simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 0.0            # 正式对弈
    self_play_temperature: float = 1.0  # 自我对弈前 30 步
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
```

### 8.2 model.py — 策略-价值双头条件残差网络

#### 8.2.1 输入特征编码（board_to_tensor）

从**当前行棋方视角**编码，保证黑白对称（同一局面交换颜色后输入张量等价）：

```
通道（input_channels=6，均为 0/1 二值）:
  [0] 己方亚马逊: grid == current_player         → 1/0
  [1] 对方亚马逊: grid == -current_player        → 1/0
  [2] 烧毁格:     grid == BURNED                 → 1/0
  [3] 空格:       grid == EMPTY                  → 1/0（冗余，但可提升收敛）
  [4] 行棋方指示: 白方行棋 → 全 1，黑方行棋 → 全 0（保留绝对颜色信息）
  [5] 最近一步标记: 上一步 from/to 标记 1.0、箭落点标记 0.5（可选，第 6 通道）
```

> 通道 [4] 使状态张量可逆（可还原绝对棋盘），供训练期重建走法列表使用；通道 [5] 可选，关闭即退化为 5 通道。

#### 8.2.2 网络结构

```
输入: (batch, input_channels, 10, 10)

  初始卷积: Conv 3×3, 256 filters → BN → ReLU
      ↓
  残差塔: 13 × ResidualBlock（AlphaGo Zero 风格）
      Conv 3×3, 256 → BN → ReLU → Conv 3×3, 256 → BN
      输出 = ReLU(x + F(x))                    # 跳跃连接
      ↓
  ┌───────────────┬─────────────────┬──────────────────┬─────────────────┐
  ▼               ▼                 ▼                  ▼                 ▼
 策略-from头    策略-to头         策略-arrow头        价值头
 Conv1×1(1)    Conv1×1(100)      Conv1×1(100)       Conv1×1(1)
 →(1,10,10)    →(100,10,10)      →(100,10,10)       → BN → ReLU
 →Flatten      空间=from 通道=to  空间=to 通道=arrow  → Flatten
 →(100,)       →P(to|from)       →P(arrow|to)       → FC(256) → ReLU
                                                     → FC(1) → Tanh → [-1,1]
```

#### 8.2.3 条件三头策略的解码

策略头采用**条件分解**（用户选定方案）：将一步走法 `(from, to, arrow)` 的联合概率分解为三因子，每个因子各自在**合法域**上做 softmax，天然满足「所有合法走法概率之和 = 1」。

```
给定网络输出:
  from_logits:  (100,)            # 起点（亚马逊所在格）得分
  to_logits:    (100, 10, 10)     # to_logits[to_idx, from_r, from_c]
  arrow_logits: (100, 10, 10)     # arrow_logits[arrow_idx, to_r, to_c]

对一步合法走法 move = (from, to, arrow):
  P(from)       = softmax(from_logits, mask=己方亚马逊格)[from_idx]
  P(to | from)  = softmax(to_logits[:, from_r, from_c],
                          mask=from 的皇后可达格)[to_idx]
  P(arrow | to) = softmax(arrow_logits[:, to_r, to_c],
                          mask=to 的皇后可达格)[arrow_idx]
  P(move)       = P(from) · P(to | from) · P(arrow | to)

「皇后可达格」由 Rules.get_queen_reachable 计算；掩码只保留合法格，
softmax 仅在掩码内的合法格上归一化。
```

> 与 v1.0 三头分解的关键差异：v1.0 用三个**独立** 100 维 softmax 再对合法走法整体重归一化，忽略了「to 依赖 from、arrow 依赖 to」的条件结构；v2.0 的条件分解让 to 头输出以 from 为条件的分布、arrow 头输出以 to 为条件的分布，策略头更贴合走法结构且天然归一化。

#### 8.2.4 接口定义

```python
class AmazonsNet(nn.Module):
    """策略-价值双头条件残差网络。"""

    def __init__(self, config: NetworkConfig | None = None):
        """按 §8.2.2 构建网络层。"""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_channels, 10, 10)

        Returns:
            {
                "policy_from":  (batch, 100)            起点 logits
                "policy_to":    (batch, 100, 10, 10)    P(to|from) logits
                "policy_arrow": (batch, 100, 10, 10)    P(arrow|to) logits
                "value":        (batch, 1)              胜率 ∈ [-1, 1]
            }
        """

    @staticmethod
    def board_to_tensor(board: Board) -> np.ndarray:
        """将 Board 转换为 (input_channels, 10, 10) numpy 数组（当前行棋方视角，见 §8.2.1）。"""

    def predict(self, board: Board) -> tuple[dict[str, np.ndarray], float]:
        """
        单局面推理（封装 board_to_tensor → forward → numpy），供 MCTS 调用。

        Returns:
            (logits_dict, value)
            value 以 board 当前行棋方视角，∈ [-1, 1]。
        """
```

#### 8.2.5 训练损失

给定一批样本 `(s, π, z)`，其中 `π` 为 MCTS 访问次数分布（联合分布），`z ∈ {-1, +1}` 为终局胜负。由 `π` 分解出条件三头的训练目标：

```
目标分解（对每个样本）:
  π_from(f)      = Σ_{a: from(a)=f} π(a)                    # from 边缘
  π_to|from(t|f) = Σ_{a: from=f, to=t} π(a) / π_from(f)     # to 条件
  π_to(t)        = Σ_{a: to(a)=t} π(a)                      # to 边缘
  π_arrow|to(x|t)= Σ_{a: to=t, arrow=x} π(a) / π_to(t)      # arrow 条件

损失:
  L_policy = -Σ_a π(a) · log P(a)
           = CE(from_logits[masked], π_from)
           + Σ_f π_from(f) · CE(to_logits[:, f][masked], π_to|from(·|f))
           + Σ_t π_to(t)   · CE(arrow_logits[:, t][masked], π_arrow|to(·|t))
  L_value  = MSE(z, v)                    # v 为 Tanh 输出
  L_total  = L_policy + L_value + weight_decay · ||θ||²
```

> **实现简化（可选）**：完整条件损失需对每个 from/to 行加权求和；首版可退化为「仅对实际落子的 from 行 / to 行计算 CE」，训练更快、更省内存，代价是梯度信号略有偏差。推荐在 GPU 充足时使用完整条件损失。

### 8.3 replay_buffer.py

```python
Sample = tuple[np.ndarray, np.ndarray, np.ndarray, float]
# (state, moves, pi, z)
#   state: (input_channels, 10, 10) int8，当前行棋方视角
#   moves: (K, 3) int16，K 个「被访问过」的合法走法 (from_idx, to_idx, arrow_idx)
#   pi:    (K,) float32，与 moves 对齐的访问次数概率（联合分布，已归一化）
#   z:     float ∈ {-1, +1}，本样本行棋方最终胜负

class ReplayBuffer:
    """经验回放池，存储自我对弈生成的训练样本。"""

    def __init__(self, capacity: int = 300000):
        """capacity: 最大样本数，超过后按 FIFO 丢弃最旧样本。"""

    def push(self, game_data: list[Sample]) -> None:
        """存入一整局棋谱的所有样本。"""

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        随机采样一个 mini-batch，返回 numpy 数组字典:
        {"state": (B, C, 10, 10), "moves": (B, K_max, 3),
         "pi": (B, K_max), "z": (B, 1)}   # K 不一，按 batch 内最大 K 补零对齐
        """

    def __len__(self) -> int:
        """当前样本总数。"""
```

> **内存估算**：每样本约 `600B(state) + K×(3×2+4)B`，K ≤ 模拟次数（800），平均更小；30 万样本约 1~2 GB，可由 `capacity` 调节。

### 8.4 selfplay.py

```python
def generate_self_play_data(
    model: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
) -> list[list[Sample]]:
    """
    使用当前网络自我对弈，生成训练数据。

    每局流程:
      1. board = Board()，mcts = MCTSEngine(model, mcts_config)
      2. 循环至终局:
         a. 温度策略: 前 30 步 temperature = self_play_temperature 且
            根节点注入 Dirichlet 噪声；之后 temperature → 0（贪心）
         b. π, move = mcts.get_policy(board, temperature)
         c. 记录 (board_to_tensor(board), 被访问走法索引, π, board.current_player)
         d. board.make_move(move)
      3. winner 确定后标注 z: 样本行棋方 == winner → +1，否则 -1
      4. 返回本局样本列表
    """

def pit_evaluate(
    new_model: AmazonsNet,
    old_model: AmazonsNet,
    num_games: int,
    mcts_config: MCTSConfig,
) -> dict[str, float]:
    """
    新网络 vs 旧网络对抗评估（temperature=0，各执一半白/黑）。

    Returns:
        {"new_win": N, "old_win": N, "new_win_rate": float}
    """
```

### 8.5 train.py

```python
def train_cycle(
    model: AmazonsNet,
    optimizer: torch.optim.Optimizer,
    scheduler: object,               # 余弦退火调度器
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
) -> dict[str, float]:
    """
    单轮训练：采样 → 前向 → 计算 L_total（§8.2.5）→ 反向 → 更新。

    Returns:
        {"policy_loss": float, "value_loss": float, "total_loss": float}
    """

def train_loop(
    model: AmazonsNet,
    config: TrainingConfig,
    mcts_config: MCTSConfig,
    device: str = "cuda",
) -> None:
    """
    完整训练闭环（需求 F-P1-05 / F-P1-06）:

      for cycle in 1..num_training_cycles:
        1. self_play_data = generate_self_play_data(model, ...)   # 自我对弈
        2. replay_buffer.push(self_play_data)                     # 入池
        3. train_cycle(...) 反复 K 步                              # 训练
        4. 若 cycle % eval_every == 0:
             pit_evaluate(new, best)
             若 new_win_rate > eval_win_rate_threshold → 替换 best
        5. 周期性 save_checkpoint(...)
    """

def save_checkpoint(model: AmazonsNet, optimizer: object, path: str, cycle: int) -> None:
    """保存模型 + 优化器 + 轮次到 .pt。"""

def load_checkpoint(path: str, config: NetworkConfig) -> AmazonsNet:
    """从 .pt 加载模型参数，返回可推理的 AmazonsNet。"""
```

### 8.6 训练闭环数据流

```
        ┌─────────────────────────────────────────────┐
        │              训练闭环 (train.py)              │
        │                                             │
        │  ① 自我对弈        ② 回放池        ③ 训练    │
        │  selfplay.py ──► replay_buffer ──► train.py │
        │  (MCTS+网络)        (样本)          (梯度更新)│
        │        ▲                                    │
        │        │  ④ 评估替换（胜率>阈值）             │
        │        └─────────── pit_evaluate ───────────┘
        │                                             │
        │  循环 ①→②→③→④ 直至 num_training_cycles 或收敛│
        └─────────────────────────────────────────────┘
```

#### 独立测试说明

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_model.py` | forward 各输出形状正确；board_to_tensor 通道语义正确；value ∈ [-1,1]；条件三头在掩码内归一化 |
| `tests/test_replay_buffer.py` | push/sample/__len__；容量满后 FIFO 淘汰 |
| `tests/test_selfplay.py` | 小规模自对弈数据格式正确；z 标注与胜负一致 |
| `tests/test_train.py` | 随机数据可反向传播、损失下降；save/load checkpoint 往返一致 |

---

## 9. main.py——程序入口

### 运行模式

程序支持四种运行模式，通过命令行参数切换：

```bash
# GUI 模式: 人类 vs AI
python main.py --mode gui --player white --engine auto

# 控制台模式: AI vs AI（观战/测试），每一步输出走法字符串
python main.py --mode console --engine mcts

# 比赛模式: 从标准输入读取对手走法，输出 AI 走法到标准输出
python main.py --mode competition --color white --time 900 --engine mcts

# 训练模式: 自我对弈 + 训练闭环（第二阶段，需 PyTorch + GPU）
python main.py --mode train
```

### 接口定义

```python
def parse_args() -> argparse.Namespace:
    """
    命令行参数:
    --mode        {gui, console, competition, train}  运行模式
    --engine      {alphabeta, mcts, auto}             搜索引擎（auto=按配置+模型可用性）
    --color       {white, black}                      比赛模式中己方颜色
    --time        int                                 比赛模式总时间（秒）
    --config      str                                 YAML 配置文件路径
    --depth       int                                 覆盖搜索深度
    --book        str                                 开局库文件路径
    --tablebase   str                                 残局库文件路径
    --model       str                                 网络 checkpoint 路径（mcts 引擎用）
    """

def setup_engine(config: dict, args) -> Engine:
    """
    工厂函数：根据配置构建搜索引擎实例（AlphaBeta 或 MCTS+网络）。

    选择逻辑（engine.type）:
      - "alphabeta":    构建 AlphaBetaEngine（兜底，见 §4.3）
      - "mcts_network": 尝试加载网络 checkpoint 并构建 MCTSEngine（§4.5 + §8）；
                        加载失败则回退 AlphaBetaEngine 并打印告警
      - "auto":         checkpoint 存在且加载成功 → MCTSEngine，否则 → AlphaBetaEngine
    """

def run_gui_mode(engine: Engine, args) -> None:
    """启动 Tkinter GUI"""

def run_console_mode(engine_white: Engine, engine_black: Engine, args) -> None:
    """
    控制台 AI vs AI 模式。
    每步输出走法字符串，显示当前棋盘（文本形式）。
    """

def run_competition_mode(engine: Engine, args) -> None:
    """
    比赛模式——标准输入/输出通信。
    
    流程:
    1. 输出 "READY" 表示就绪
    2. 若己方执白，先搜索并输出走法
    3. 循环:
       a. 从 stdin 读取一行 → 对手走法字符串
       b. 解析为 Move → board.make_move()
       c. 搜索己方走法
       d. 输出走法字符串到 stdout
    4. 终局时输出 "RESIGN" 或自动结束
    """

def run_training_mode(args) -> None:
    """
    训练模式：加载模型 → 运行自我对弈 + 训练闭环（见 §8.5 train_loop）。
    仅在第二阶段启用，需 PyTorch + GPU。
    """
    from src.network.config import MCTSConfig, NetworkConfig, TrainingConfig
    from src.network.model import AmazonsNet
    from src.network.train import train_loop

    # 按 config.yaml 的 training/network/mcts 段构建配置并调用 train_loop(...)


if __name__ == "__main__":
    args = parse_args()
    engine = setup_engine(args)
    
    if args.mode == "gui":
        run_gui_mode(engine, args)
    elif args.mode == "console":
        run_console_mode(engine, engine, args)  # 控制台两边用同一引擎
    elif args.mode == "competition":
        run_competition_mode(engine, args)
    elif args.mode == "train":
        run_training_mode(args)
```

### 比赛模式时序图

```
程序 (stdout)                    人类操作员                   比赛平台
     │                              │                          │
     │──── "READY" ────────────────►│                          │
     │                              │                          │
     │  [若执白]                     │                          │
     │──── "d1-d5-g8" ─────────────►│                          │
     │                              │──── 手动落子 ────────────►│
     │                              │                          │
     │                              │◄─── 对手走法显示 ────────│
     │◄── 输入对手走法字符串 ───────│                          │
     │                              │                          │
     │──── "g10-g6-c10" ───────────►│                          │
     │                              │──── 手动落子 ────────────►│
     │                              │                          │
     │          ... 循环至终局 ...                              │
```

---

## 10. 配置文件设计

### config.yaml 完整结构

```yaml
# ============================================================
# 亚马逊棋博弈系统 - 全局配置文件
# ============================================================

# --- 游戏规则 ---
game:
  board_size: 10
  num_amazons: 4

# --- 引擎选择 ---
engine:
  type: "auto"                    # "alphabeta" | "mcts_network" | "auto"
                                  # auto: 模型可用则 MCTS，否则回退 Alpha-Beta

# --- Alpha-Beta 搜索（兜底方案） ---
alphabeta:
  max_depth: 5                    # 最大搜索深度
  time_budget_ms: 15000           # 单步最大时间 (ms)
  use_transposition_table: true   # 是否启用置换表
  use_iterative_deepening: true   # 是否启用迭代加深
  use_move_ordering: true         # 是否启用走法排序
  transposition_table_size: 1048576  # 2^20

# --- 局面评估（Alpha-Beta 叶节点用） ---
evaluator:
  type: "heuristic"               # "heuristic"（快速）或 "monte_carlo"（慢）
  num_simulations: 100            # 仅 monte_carlo 模式使用

# --- MCTS 搜索 ---
mcts:
  num_simulations: 800            # 每次搜索的模拟次数
  c_puct: 1.0                     # PUCT 探索常数
  temperature: 0.0                # 走子温度 (0 = 贪心)

# --- 网络模型 ---
network:
  model_path: "models/best_model.pt"
  device: "cuda"                  # "cuda" | "cpu"
  input_channels: 6
  num_residual_blocks: 13
  num_filters: 256

# --- 训练 ---
training:
  batch_size: 512
  learning_rate: 0.01
  weight_decay: 1e-4
  training_steps_per_cycle: 1000
  replay_buffer_capacity: 300000  # 最大样本数
  self_play_games_per_cycle: 200
  eval_games: 100
  eval_win_rate_threshold: 0.55
  num_training_cycles: 100

# --- 开局库 ---
opening_book:
  enabled: false
  book_path: "src/opening_book/data/opening_book.json"

# --- 残局库 ---
endgame:
  enabled: false                  # 初赛先关闭，预计算耗时
  tablebase_path: "src/endgame/data/tablebase.pkl"
  max_pieces: 3                   # 残局库预计算范围: 双方共 ≤3 子
  max_empty_squares: 15           # 可走空格数 ≤15

# --- GUI ---
gui:
  cell_size: 60
  piece_radius: 22
  board_color: "#DEB887"
  default_mode: "HUMAN_VS_AI"     # HUMAN_VS_AI | AI_VS_HUMAN | AI_VS_AI

# --- 日志 ---
logging:
  level: "INFO"                   # DEBUG | INFO | WARNING | ERROR
  game_log_enabled: true
  game_log_path: "logs/games/"
  search_stats_enabled: true      # 是否输出搜索统计
```

---

## 11. 测试计划

### 11.1 测试层次

```
┌─────────────────────────────────────────────┐
│           集成测试 (Integration)              │
│   AI vs Random 100 局, AI vs AI 对弈         │
├─────────────────────────────────────────────┤
│           单元测试 (Unit)                     │
│   每个 .py 文件对应一个 test_*.py             │
├─────────────────────────────────────────────┤
│           冒烟测试 (Smoke)                    │
│   启动即退、走法生成不崩溃、GUI 可打开        │
└─────────────────────────────────────────────┘
```

### 11.2 单元测试清单

| 测试文件 | 被测模块 | 关键测试用例 | 预期结果 |
|----------|----------|-------------|----------|
| `tests/test_move.py` | `game/move.py` | `from_string("d10-d5-g7")` 往返 | `move.to_string() == "d10-d5-g7"` |
| | | `from_string("")` 非法输入 | `ValueError` |
| | | `from_string("a4-d4")` 缺箭段落 | `ValueError` |
| | | "a1" ↔ (9,0), "j10" ↔ (0,9) | 边界映射正确 |
| `tests/test_board.py` | `game/board.py` | 初始棋盘 4+4 子在正确位置 | 位置匹配 `INITIAL_*_POSITIONS` |
| | | `make_move` → 棋子移动、箭格烧毁、玩家切换 | 三项均正确 |
| | | `make_move` + `undo_move` 往返 10 次 | 棋盘完全恢复 |
| | | `clone` 后修改原棋盘 | 克隆不受影响 |
| | | Zobrist 哈希一致性 | 同一局面两次到达哈希相同 |
| `tests/test_rules.py` | `game/rules.py` | 初始棋盘合法走法总数 | 手工计算参考值 ±5% |
| | | 终局局面 `is_game_over()` | 返回 True |
| | | 中心空旷位置 `get_queen_reachable` | 返回 27 格（8方向全通） |
| | | 有烧毁格阻塞的走法生成 | 不穿过烧毁格 |
| `tests/test_evaluator.py` | `ai/evaluator.py` | 对手无合法走法局面 | `evaluate()` 返回 1.0 |
| | | 己方无合法走法局面 | `evaluate()` 返回 0.0 |
| | | 固定随机种子，多次调用 | 返回值一致 |
| `tests/test_alphabeta.py` | `ai/alphabeta.py` | 构造"一步杀"局面 | 搜索返回必胜走法 |
| | | 构造"必败"局面（无合法走法） | 搜索返回保底走法 |
| | | 限制 0.1 秒超时 | 仍返回合法走法 |
| | | depth=2, 3, 4 搜索结果一致性 | 更深不更差（统计检验） |
| `tests/test_transposition.py` | `ai/transposition.py` | 同一局面哈希一致 | 相等 |
| | | 不同局面哈希不同 | 不相等（碰撞率低） |
| | | `store` → `lookup` 往返 | 取回值匹配 |
| | | `clear` 后 | `lookup` 返回 None |
| `tests/test_opening_book.py` | `opening_book/book.py` | 加载示例 JSON | 不抛异常 |
| | | 命中查询 | 返回候选列表 |
| | | 未命中查询 | 返回 None |
| `tests/test_endgame.py` | `endgame/tablebase.py` | 加载预计算 .pkl | 不抛异常 |
| | | 命中查询 | 返回正确 EndgameResult |
| `tests/test_mcts.py` | `ai/mcts.py` | 搜索结果合法 | 走法 ∈ 合法走法集 |
| | | 一步必胜局面 | 搜索找到杀招 |
| | | 终局局面 | 抛 ValueError 或返回 None |
| | | `PureMCTSEngine` | 返回合法走法 |
| `tests/test_model.py` | `network/model.py` | forward 输出形状 | 与接口定义一致 |
| | | board_to_tensor 通道 | 己/对/烧/空/行棋方语义正确 |
| | | 价值输出 | value ∈ [-1, 1] |
| `tests/test_replay_buffer.py` | `network/replay_buffer.py` | push/sample/__len__ | 采样形状正确 |
| | | 容量满 | FIFO 淘汰最旧样本 |
| `tests/test_selfplay.py` | `network/selfplay.py` | 小规模自对弈 | 数据格式正确 |
| | | z 标注 | 与终局胜负一致 |
| `tests/test_train.py` | `network/train.py` | 随机数据训练 | 损失可反向传播、下降 |
| | | checkpoint 往返 | save → load 参数一致 |

### 11.3 集成测试

```bash
# 冒烟测试: 程序能启动并退出
python main.py --mode console --ai-white --ai-black --depth 1  # 1步即退

# 完整对弈测试: AlphaBeta vs Random, 100 局
python -m tests.test_match --engine alphabeta --opponent random --games 100
# 预期: AlphaBeta 胜率 > 90%

# AI vs AI 测试: 验证每一步输出的走法合法性
python -m tests.test_match --engine alphabeta --opponent alphabeta --games 10
# 预期: 每步走法均为合法走法, 对局终局时胜者判定正确
```

### 11.4 测试运行命令

```bash
# 运行全部单元测试
python -m pytest tests/ -v

# 运行单个模块测试
python -m pytest tests/test_board.py -v

# 带覆盖率报告
python -m pytest tests/ --cov=src --cov-report=term
```

---

> **版本记录**  
> - v1.0 (2026-07-18): 初版，涵盖第一阶段所有模块的详细设计，第二阶段仅接口定义。
> - v2.0 (2026-08-18): 升级第二阶段为完全体 MCTS+ 深度网络完整设计——新增 §4.5 MCTS 引擎、重写 §8 网络模块（条件三头残差网络 + 自对弈 + 训练闭环）、更新 §9/§10/§11。
