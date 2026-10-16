"""亚马逊棋全局常量与坐标转换函数。

本模块定义棋盘尺寸、格位编码、初始布局、皇后走法等全局常量，
并提供代数记谱法与内部坐标之间的双向转换。
"""

# =============================================================================
# 棋盘尺寸
# =============================================================================

BOARD_SIZE: int = 10  # 10×10 棋盘

# =============================================================================
# 格位编码
# =============================================================================

EMPTY: int = 0        # 空格
WHITE: int = 1        # 白方亚马逊
BLACK: int = -1       # 黑方亚马逊
BURNED: int = 2       # 烧毁格（障碍物）

# =============================================================================
# 胜负常量
# =============================================================================

WIN: int = 1
LOSE: int = -1
DRAW: int = 0   # 理论上亚马逊棋无平局，保留以备扩展

# =============================================================================
# 皇后走法 8 方向
# =============================================================================

QUEEN_DIRECTIONS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# =============================================================================
# 代数记谱法映射
# =============================================================================

COL_LABELS: str = "abcdefghij"       # col 0→'a', col 9→'j'
# 行号: row 0→'10', row 9→'1'  (内部 row 0 是第 10 行, row 9 是第 1 行)

# =============================================================================
# 初始布局
# =============================================================================

# 黑方 (棋盘上方: d10, g10, a7, j7)
# row: 0=第10行, 3=第7行
INITIAL_BLACK_POSITIONS: list[tuple[int, int]] = [
    (0, 3),   # d10
    (0, 6),   # g10
    (3, 0),   # a7
    (3, 9),   # j7
]

# 白方 (棋盘下方: d1, g1, a4, j4)
# row: 6=第4行, 9=第1行
INITIAL_WHITE_POSITIONS: list[tuple[int, int]] = [
    (6, 0),   # a4
    (6, 9),   # j4
    (9, 3),   # d1
    (9, 6),   # g1
]

# =============================================================================
# 坐标转换函数
# =============================================================================


def pos_to_algebraic(row: int, col: int) -> str:
    """将内部坐标 (row, col) 转换为代数记谱法字符串。

    Args:
        row: 行号 (0-9), 0 对应第 10 行, 9 对应第 1 行
        col: 列号 (0-9), 0 对应 a 列, 9 对应 j 列

    Returns:
        代数记谱法字符串，如 "d10", "g1"

    Raises:
        ValueError: 坐标越界

    Examples:
        >>> pos_to_algebraic(0, 3)
        'd10'
        >>> pos_to_algebraic(9, 6)
        'g1'
    """
    if not is_in_bounds(row, col):
        raise ValueError(f"坐标越界: ({row}, {col})")
    col_str: str = COL_LABELS[col]
    # row 0 → '10', row 1→'9', ..., row 9→'1'
    row_str: str = str(10 - row)
    return f"{col_str}{row_str}"


def algebraic_to_pos(notation: str) -> tuple[int, int]:
    """将代数记谱法字符串转换为内部坐标 (row, col)。

    列号 "a"-"j" 为单字符（固定），行号 "1"-"10" 可能为单字符或双字符。
    解析策略：首字符为列号，剩余字符为行号。

    Args:
        notation: 代数记谱法字符串，如 "d10", "g1", "a4"

    Returns:
        (row, col) 元组

    Raises:
        ValueError: 记谱法格式非法或坐标越界

    Examples:
        >>> algebraic_to_pos("d10")
        (0, 3)
        >>> algebraic_to_pos("g1")
        (9, 6)
    """
    if len(notation) < 2 or len(notation) > 3:
        raise ValueError(f"非法记谱法格式: '{notation}'")

    col_char: str = notation[0]
    row_str: str = notation[1:]

    if col_char not in COL_LABELS:
        raise ValueError(f"非法列号: '{col_char}'")

    col: int = COL_LABELS.index(col_char)

    try:
        row_num = int(row_str)
    except ValueError:
        raise ValueError(f"非法行号: '{row_str}'")

    if row_num < 1 or row_num > BOARD_SIZE:
        raise ValueError(f"行号越界: {row_num}")

    # row_num 1→9, row_num 10→0
    row: int = BOARD_SIZE - row_num

    if not is_in_bounds(row, col):
        raise ValueError(f"坐标越界: ({row}, {col}) from '{notation}'")

    return (row, col)


def is_in_bounds(row: int, col: int) -> bool:
    """判断 (row, col) 是否在棋盘范围内。

    Args:
        row: 行号
        col: 列号

    Returns:
        True 如果在 [0, BOARD_SIZE) 范围内
    """
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
