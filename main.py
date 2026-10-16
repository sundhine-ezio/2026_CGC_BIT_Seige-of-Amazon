"""亚马逊棋计算机博弈系统——程序入口。

支持四种运行模式:
  - gui:         图形界面（人类 vs AI / AI vs AI 观战）
  - console:     控制台 AI vs AI 文本输出
  - competition: 比赛模式（stdin/stdout 通信）
  - train:       训练模式（自我对弈 → 训练 → 对抗评估闭环）

搜索引擎可通过 --engine 选择 alpha-beta / MCTS（auto 自动选择）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from src.ai.alphabeta import AlphaBetaEngine
from src.ai.engine import Engine
from src.ai.evaluator import HeuristicEvaluator, MonteCarloEvaluator
from src.ai.mcts import MCTSEngine
from src.config_loader import load_config
from src.game.board import Board
from src.game.move import Move
from src.game.rules import Rules

# =============================================================================
# 命令行参数
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="亚马逊棋计算机博弈系统"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gui",
        choices=["gui", "console", "competition", "train"],
        help="运行模式 (默认: gui)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["alphabeta", "mcts", "auto"],
        help="搜索引擎类型 (默认: auto；auto 在有模型时用 MCTS，否则 alpha-beta)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="MCTS 网络模型权重路径 (.pt) (默认: 使用 config.yaml 的 network.model_path)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="比赛模式中己方颜色 (默认: white)",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=900,
        help="比赛模式总时间（秒）(默认: 900)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="YAML 配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="覆盖搜索深度 (0=使用配置文件设置)",
    )
    parser.add_argument(
        "--book",
        type=str,
        default="",
        help="开局库文件路径",
    )
    parser.add_argument(
        "--tablebase",
        type=str,
        default="",
        help="残局库文件路径",
    )
    return parser.parse_args()


# =============================================================================
# 引擎工厂
# =============================================================================


def setup_engine(config: dict[str, Any], args: argparse.Namespace) -> Engine:
    """根据配置构建搜索引擎实例（Engine 抽象，与 alpha-beta / MCTS 解耦）。

    engine 类型:
      - alphabeta: 启发式 + alpha-beta 搜索
      - mcts:      网络引导 MCTS（需模型权重 + PyTorch）
      - auto:      有可用模型 → MCTS；否则回退 alpha-beta

    Args:
        config: 配置字典
        args: 命令行参数

    Returns:
        Engine 实例
    """
    engine_type: str = str(args.engine or config.get("engine", {}).get("type", "auto"))

    if engine_type == "mcts":
        return _setup_mcts_engine(config, args)
    if engine_type == "alphabeta":
        return _setup_alphabeta_engine(config, args)

    # auto：优先 MCTS（有模型时），否则回退 alpha-beta
    try:
        return _setup_mcts_engine(config, args)
    except Exception as e:
        print(f"[INFO] MCTS 不可用（{e}），回退 alpha-beta", file=sys.stderr)
        return _setup_alphabeta_engine(config, args)


def _setup_alphabeta_engine(config: dict[str, Any], args: argparse.Namespace) -> AlphaBetaEngine:
    """构建 AlphaBetaEngine（启发式 + alpha-beta 搜索）。"""
    ab_config: dict[str, Any] = config.get("alphabeta", {})
    eval_config: dict[str, Any] = config.get("evaluator", {})

    max_depth: int = args.depth if args.depth > 0 else ab_config.get("max_depth", 5)
    time_budget_ms: float = float(ab_config.get("time_budget_ms", 15000))
    use_move_ordering: bool = bool(ab_config.get("use_move_ordering", True))

    # 根据配置选择评估器类型
    eval_type: str = str(eval_config.get("type", "heuristic"))
    if eval_type == "monte_carlo":
        num_simulations: int = int(eval_config.get("num_simulations", 100))
        evaluator: HeuristicEvaluator | MonteCarloEvaluator = MonteCarloEvaluator(
            num_simulations=num_simulations
        )
    else:
        evaluator = HeuristicEvaluator()

    engine: AlphaBetaEngine = AlphaBetaEngine(
        evaluator=evaluator,
        max_depth=max_depth,
        time_budget_ms=time_budget_ms,
        use_transposition_table=bool(
            ab_config.get("use_transposition_table", True)
        ),
        use_iterative_deepening=bool(
            ab_config.get("use_iterative_deepening", True)
        ),
        use_move_ordering=use_move_ordering,
        transposition_table_size=int(
            ab_config.get("transposition_table_size", 2**20)
        ),
    )

    # 可选: 注入开局库
    book_config: dict[str, Any] = config.get("opening_book", {})
    if book_config.get("enabled", False) or args.book:
        book_path: str = args.book or str(book_config.get("book_path", ""))
        if book_path:
            try:
                from src.opening_book.book import OpeningBook

                engine.set_opening_book(OpeningBook(book_path))
            except Exception as e:
                print(f"[WARN] 开局库加载失败: {e}", file=sys.stderr)

    # 可选: 注入残局库
    endgame_config: dict[str, Any] = config.get("endgame", {})
    if endgame_config.get("enabled", False) or args.tablebase:
        tb_path: str = args.tablebase or str(endgame_config.get("tablebase_path", ""))
        if tb_path:
            try:
                from src.endgame.tablebase import Tablebase

                engine.set_endgame_tablebase(Tablebase(tb_path))
            except Exception as e:
                print(f"[WARN] 残局库加载失败: {e}", file=sys.stderr)

    return engine


def _resolve_model_path(config: dict[str, Any], args: argparse.Namespace) -> str:
    """解析模型权重路径：命令行 > config.yaml network.model_path。"""
    network_cfg: dict[str, Any] = config.get("network", {})
    engine_cfg: dict[str, Any] = config.get("engine", {})
    return args.model or str(network_cfg.get("model_path", "") or engine_cfg.get("model_path", ""))


def _setup_mcts_engine(config: dict[str, Any], args: argparse.Namespace) -> MCTSEngine:
    """构建网络引导 MCTS 引擎（需模型权重 + PyTorch）。"""
    model_path: str = _resolve_model_path(config, args)
    if not model_path:
        raise RuntimeError("MCTS 引擎需要模型权重路径（--model 或 config.yaml network.model_path）")

    if not os.path.isfile(model_path):
        raise RuntimeError(f"模型权重文件不存在: {model_path}")

    # 延迟导入 torch 相关模块，避免 alpha-beta 纯模式强依赖 torch
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError("MCTS 引擎需要 PyTorch，请先安装 torch") from e

    from src.network.config import MCTSConfig, NetworkConfig
    from src.network.train import load_checkpoint

    network_cfg: dict[str, Any] = config.get("network", {})
    mcts_cfg: dict[str, Any] = config.get("mcts", {})

    net_config = NetworkConfig(
        input_channels=int(network_cfg.get("input_channels", 6)),
        num_residual_blocks=int(network_cfg.get("num_residual_blocks", 13)),
        num_filters=int(network_cfg.get("num_filters", 256)),
        board_feature_dim=int(network_cfg.get("board_feature_dim", 256)),
        move_embedding_dim=int(network_cfg.get("move_embedding_dim", 128)),
        hidden_dim=int(network_cfg.get("hidden_dim", 128)),
    )
    model = load_checkpoint(model_path, net_config)

    mcts_config = MCTSConfig(
        num_simulations=int(mcts_cfg.get("num_simulations", 800)),
        c_puct=float(mcts_cfg.get("c_puct", 1.0)),
        temperature=float(mcts_cfg.get("temperature", 0.0)),
    )
    return MCTSEngine(
        model,
        num_simulations=mcts_config.num_simulations,
        c_puct=mcts_config.c_puct,
        temperature=mcts_config.temperature,
    )


# =============================================================================
# 运行模式
# =============================================================================


def run_gui_mode(engine: Engine, config: dict[str, Any]) -> None:
    """启动 Tkinter GUI（对局模式由启动时的对话框决定）。

    Args:
        engine: AI 搜索引擎
        config: 配置字典
    """
    from src.gui.main_window import AmazonsGUI

    gui: AmazonsGUI = AmazonsGUI(engine)
    gui.run()


def _print_board_text(board: Board) -> None:
    """以文本形式打印当前棋盘。"""
    cols: str = "   a  b  c  d  e  f  g  h  i  j"
    print(cols)
    print("  +" + "---" * 10 + "+")

    grid = board.get_grid()
    for r in range(10):
        row_label: str = str(10 - r).rjust(2)
        cells: list[str] = []
        for c in range(10):
            val: int = int(grid[r, c])
            if val == 1:
                cells.append("●")
            elif val == -1:
                cells.append("○")
            elif val == 2:
                cells.append("×")
            else:
                cells.append("·")
        print(f"{row_label} | {'  '.join(cells)} |")

    print("  +" + "---" * 10 + "+")


def run_console_mode(engine_white: Engine, engine_black: Engine) -> None:
    """控制台 AI vs AI 模式。

    Args:
        engine_white: 白方引擎
        engine_black: 黑方引擎
    """
    board: Board = Board()
    move_count: int = 0

    print("=" * 50)
    print("亚马逊棋 AI vs AI 对弈")
    print("=" * 50)

    while not Rules.is_game_over(board):
        _print_board_text(board)

        current_player: str = "白方" if board.get_current_player() == 1 else "黑方"
        engine = engine_white if board.get_current_player() == 1 else engine_black

        print(f"\n[{current_player}] 搜索中...")
        time_budget: float = 15000.0
        move: Move = engine.search(board.clone(), time_budget)

        print(f"[{current_player}] {move.to_string()}")
        board.make_move(move)
        move_count += 1

    _print_board_text(board)
    winner: int = Rules.get_winner(board)
    winner_name: str = "白方" if winner == 1 else "黑方"
    print(f"\n{'=' * 50}")
    print(f"  游戏结束！{winner_name} 获胜！  总步数: {move_count}")
    print(f"{'=' * 50}")


# =============================================================================
# 比赛模式时间分配
# =============================================================================

# 总时间安全系数：整局最多用掉预算的这个比例
COMPETITION_SAFETY: float = 0.8
# 预估整局步数（亚马逊棋通常 60~120 步），用于把剩余时间均摊到剩余步数
COMPETITION_EXPECTED_MOVES: int = 100
# 剩余步数下限，防止末盘（战局拖长时）每步时间反而反弹
COMPETITION_MIN_MOVES_LEFT: int = 20
# 传给引擎的最小预算。**不可为 0**：AlphaBetaEngine 把 <=0 解释为"不限时"，
# 会导致单步无限搜索直至超时判负。
COMPETITION_MIN_MOVE_MS: float = 1.0


def _competition_move_budget_ms(
    elapsed_ms: float, moves_played: int, total_time_ms: float
) -> float:
    """按剩余时间与预估剩余步数，动态分配本步搜索预算。

    采用「剩余时间固定比例」策略：每步至多花掉剩余时间的
    COMPETITION_SAFETY / COMPETITION_MIN_MOVES_LEFT，因此即使实际步数
    远超预估，累计用时也不会超过总预算（几何级数收敛）。

    Args:
        elapsed_ms: 本局已用时间（毫秒）
        moves_played: 本局已下步数
        total_time_ms: 本局总时间预算（毫秒）

    Returns:
        本步搜索时间预算（毫秒），恒为正数
    """
    remaining_ms: float = total_time_ms - elapsed_ms
    if remaining_ms <= 0:
        # 已超预算：只给最小预算尽快出一个合法走法，避免拖延
        return COMPETITION_MIN_MOVE_MS

    moves_left: int = max(
        COMPETITION_EXPECTED_MOVES - moves_played, COMPETITION_MIN_MOVES_LEFT
    )
    budget_ms: float = remaining_ms * COMPETITION_SAFETY / moves_left
    # 单步预算不超过剩余时间 → 累计用时不可能超过总预算；下限保证不为 0
    return min(max(budget_ms, COMPETITION_MIN_MOVE_MS), remaining_ms)


def run_competition_mode(engine: Engine, args: argparse.Namespace) -> None:
    """比赛模式——标准输入输出通信。

    协议:
      1. 输出 "READY" 表示就绪
      2. 若己方执白，先搜索并输出走法
      3. 循环: 读对手走法 → 搜索 → 输出己方走法

    每步时间预算按剩余时间动态分配（见 _competition_move_budget_ms），
    保证整局总用时不超过 --time 指定的秒数。

    Args:
        engine: AI 搜索引擎
        args: 命令行参数
    """
    board: Board = Board()
    my_color: int = 1 if args.color == "white" else -1  # WHITE=1, BLACK=-1
    total_time_ms: float = float(args.time * 1000)
    start_time: float = time.time()
    moves_played: int = 0

    print("READY", flush=True)

    # 若己方执白 → 先走
    if my_color == 1:
        move: Move = engine.search(
            board.clone(),
            _competition_move_budget_ms(
                (time.time() - start_time) * 1000.0, moves_played, total_time_ms
            ),
        )
        print(move.to_string(), flush=True)
        board.make_move(move)
        moves_played += 1

    # 主循环
    while not Rules.is_game_over(board):
        line: str = sys.stdin.readline().strip()
        if not line or line.upper() == "RESIGN":
            break

        try:
            opponent_move: Move = Move.from_string(line)
        except ValueError:
            print(f"ERROR: 非法走法格式: '{line}'", file=sys.stderr, flush=True)
            continue

        # 校验走法确实符合规则。格式合法但不合法的走法（例如箭落在自己落脚的
        # 格子上）会污染棋盘内部状态，使后续搜索抛 KeyError 直接崩掉程序，
        # 比赛中操作员输错一行即致命，必须在此拦下。
        if opponent_move not in set(Rules.get_legal_moves(board)):
            print(
                f"ERROR: 非法走法（不符合规则）: '{line}'", file=sys.stderr, flush=True
            )
            continue

        board.make_move(opponent_move)

        if Rules.is_game_over(board):
            print("RESIGN", flush=True)
            break

        my_move: Move = engine.search(
            board.clone(),
            _competition_move_budget_ms(
                (time.time() - start_time) * 1000.0, moves_played, total_time_ms
            ),
        )
        print(my_move.to_string(), flush=True)
        board.make_move(my_move)
        moves_played += 1

        if Rules.is_game_over(board):
            break


# =============================================================================
# 训练模式
# =============================================================================


def run_train_mode(config: dict[str, Any], args: argparse.Namespace) -> None:
    """训练模式：启动完整「自我对弈 → 训练 → 对抗评估 → 替换」闭环。

    Args:
        config: 配置字典
        args: 命令行参数（--model 可用于从已有 checkpoint 继续训练）
    """
    from src.network.config import MCTSConfig, NetworkConfig, TrainingConfig
    from src.network.model import AmazonsNet
    from src.network.train import load_checkpoint, train_loop

    network_cfg: dict[str, Any] = config.get("network", {})
    mcts_cfg: dict[str, Any] = config.get("mcts", {})
    training_cfg: dict[str, Any] = config.get("training", {})

    net_config = NetworkConfig(
        input_channels=int(network_cfg.get("input_channels", 6)),
        num_residual_blocks=int(network_cfg.get("num_residual_blocks", 13)),
        num_filters=int(network_cfg.get("num_filters", 256)),
        board_feature_dim=int(network_cfg.get("board_feature_dim", 256)),
        move_embedding_dim=int(network_cfg.get("move_embedding_dim", 128)),
        hidden_dim=int(network_cfg.get("hidden_dim", 128)),
    )
    mcts_config = MCTSConfig(
        num_simulations=int(mcts_cfg.get("num_simulations", 800)),
        c_puct=float(mcts_cfg.get("c_puct", 1.0)),
        temperature=float(mcts_cfg.get("temperature", 0.0)),
        self_play_temperature=float(mcts_cfg.get("self_play_temperature", 1.0)),
        dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
        dirichlet_epsilon=float(mcts_cfg.get("dirichlet_epsilon", 0.25)),
    )
    train_config = TrainingConfig.from_dict(training_cfg)

    # 可选：从已有 checkpoint 继续训练
    model_path: str = _resolve_model_path(config, args)
    if model_path and os.path.isfile(model_path):
        model = load_checkpoint(model_path, net_config)
        print(f"[train] 从 {model_path} 继续训练", flush=True)
    else:
        model = AmazonsNet(net_config)
        if model_path:
            print(f"[WARN] 模型文件不存在，从头训练: {model_path}", file=sys.stderr)

    device: str = str(training_cfg.get("device", "cuda"))
    train_loop(model, train_config, mcts_config, device=device)


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    """程序主入口。"""
    args: argparse.Namespace = parse_args()

    try:
        config: dict[str, Any] = load_config(args.config)
    except (FileNotFoundError, Exception) as e:
        print(f"[WARN] 配置文件加载失败 ({e})，使用默认设置", file=sys.stderr)
        config = {}

    if args.mode == "train":
        run_train_mode(config, args)
        return

    engine: Engine = setup_engine(config, args)

    if args.mode == "gui":
        run_gui_mode(engine, config)
    elif args.mode == "console":
        run_console_mode(engine, engine)  # 两边使用同一引擎
    elif args.mode == "competition":
        run_competition_mode(engine, args)


if __name__ == "__main__":
    main()
