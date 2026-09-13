"""主 GUI 窗口。

基于 Tkinter 的亚马逊棋图形界面，支持人机对弈、AI vs AI 观战。
走法输入使用三阶段状态机: IDLE → AMAZON_SELECTED → MOVE_SELECTED → IDLE
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

from src.ai.engine import Engine
from src.game.board import Board
from src.game.constants import BLACK, EMPTY, WHITE
from src.game.move import Move
from src.game.rules import Rules
from src.gui.board_painter import BoardCanvas
from src.gui.dialogs import (
    show_error_dialog,
    show_game_over_dialog,
    show_mode_selection_dialog,
)


class AmazonsGUI:
    """亚马逊棋主 GUI 窗口。"""

    # 输入状态机常量
    IDLE: str = "IDLE"
    AMAZON_SELECTED: str = "AMAZON_SELECTED"
    MOVE_SELECTED: str = "MOVE_SELECTED"

    def __init__(self, engine: Engine) -> None:
        """初始化 GUI（不设置对局模式——由启动/新局时的对话框决定）。

        Args:
            engine: AI 搜索引擎实例
        """
        self._engine: Engine = engine
        self._board: Board = Board()

        # 对局模式（由 show_mode_selection_dialog 设置）
        self._white_player: str = "human"
        self._black_player: str = "ai"

        # 输入状态机
        self._input_state: str = self.IDLE
        self._selected_amazon: tuple[int, int] | None = None
        self._selected_move_target: tuple[int, int] | None = None

        # AI 控制
        self._ai_thinking: bool = False

        # 走法记录
        self._move_count: int = 0

        # 构建 UI
        self._root: tk.Tk = tk.Tk()
        self._root.title("亚马逊棋 - 计算机博弈大赛")
        self._root.resizable(False, False)

        # ---- 主布局: 棋盘 (左) + 走法记录 (右) ----
        main_frame: tk.Frame = tk.Frame(self._root)
        main_frame.pack(pady=5, padx=5)

        self._board_canvas: BoardCanvas = BoardCanvas(main_frame)
        self._board_canvas.set_click_callback(self._on_board_click)
        self._board_canvas.pack(side=tk.LEFT)

        # 右侧走法记录面板
        log_frame: tk.Frame = tk.Frame(main_frame)
        log_frame.pack(side=tk.LEFT, padx=(10, 0), fill=tk.BOTH)

        tk.Label(log_frame, text="走法记录", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(0, 3)
        )

        log_list_frame: tk.Frame = tk.Frame(log_frame)
        log_list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar: tk.Scrollbar = tk.Scrollbar(log_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._move_log: tk.Listbox = tk.Listbox(
            log_list_frame,
            width=26,
            height=22,
            font=("Courier New", 10),
            yscrollcommand=scrollbar.set,
        )
        self._move_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._move_log.yview)

        # 状态栏
        self._status_label: tk.Label = tk.Label(
            self._root, text="", font=("Arial", 12)
        )
        self._status_label.pack(pady=2)

        # 按钮栏
        btn_frame: tk.Frame = tk.Frame(self._root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="新局", command=self._on_new_game, width=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="悔棋", command=self._on_undo, width=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="退出", command=self._root.quit, width=8).pack(
            side=tk.LEFT, padx=5
        )

    # =========================================================================
    # 公共接口
    # =========================================================================

    def run(self) -> None:
        """启动 GUI 主循环——先弹出模式选择，再开始对局。"""
        self._show_mode_dialog_and_start()
        self._root.mainloop()

    # =========================================================================
    # 模式选择
    # =========================================================================

    def _show_mode_dialog_and_start(self) -> None:
        """弹出对局模式选择对话框，然后启动新对局。"""
        mode: dict[str, str] | None = show_mode_selection_dialog(self._root)
        if mode is None:
            # 用户取消 → 使用默认模式
            mode = {"white": "human", "black": "ai"}

        self._white_player = mode["white"]
        self._black_player = mode["black"]
        self._start_game()

    def _start_game(self) -> None:
        """根据当前模式设置启动/重启对局。"""
        self._board.reset()
        self._engine.reset()
        self._reset_input()
        self._board_canvas.set_last_move(None)  # type: ignore[arg-type]
        self._ai_thinking = False
        self._move_count = 0
        self._move_log.delete(0, tk.END)
        self._update_display()

        # 标题更新
        w: str = "人类" if self._white_player == "human" else "AI"
        b: str = "人类" if self._black_player == "human" else "AI"
        self._root.title(f"亚马逊棋 - 白方({w}) vs 黑方({b})")

        # 若 AI 先行或双方都是 AI → 触发 AI
        if self._white_player == "ai":
            self._root.after(500, self._on_ai_move)
        elif self._white_player == "human" and self._black_player == "ai":
            # 人类执白先行
            self._update_status("当前: 白方（你的回合）")

    # =========================================================================
    # 走法输入状态机
    # =========================================================================

    def _on_board_click(self, row: int, col: int) -> None:
        """棋盘点击事件处理——三阶段状态机。"""
        if self._ai_thinking:
            return

        current: int = self._board.get_current_player()
        if not self._is_human_player(current):
            return  # 非人类回合

        clicked_val: int = self._board.get_piece_at(row, col)

        if self._input_state == self.IDLE:
            # 点击己方亚马逊 → 选中
            if clicked_val == current:
                self._selected_amazon = (row, col)
                self._input_state = self.AMAZON_SELECTED
                player_name: str = "白方" if current == WHITE else "黑方"
                self._update_status(f"已选 {player_name} 子 → 请点击移动目标")

        elif self._input_state == self.AMAZON_SELECTED:
            # 点击己方另一亚马逊 → 切换
            if clicked_val == current:
                self._selected_amazon = (row, col)
                self._update_status("切换选择 → 请点击移动目标")
            # 点击空格（合法移动目标）→ 确认移动目标
            elif clicked_val == EMPTY and self._is_legal_move_target(row, col):
                self._selected_move_target = (row, col)
                self._input_state = self.MOVE_SELECTED
                self._update_status("移动目标已选 → 请点击箭落点")
            else:
                show_error_dialog(self._root, "非法的移动目标！")
                self._reset_input()

        elif self._input_state == self.MOVE_SELECTED:
            # 点击空格（合法箭落点）→ 完整走法
            if clicked_val == EMPTY and self._is_legal_arrow_target(row, col):
                move: Move = Move(
                    from_pos=self._selected_amazon,  # type: ignore[arg-type]
                    to_pos=self._selected_move_target,  # type: ignore[arg-type]
                    arrow_pos=(row, col),
                )
                self._execute_human_move(move)
                self._reset_input()
            else:
                show_error_dialog(self._root, "非法的箭落点！")
                self._reset_input()

    # =========================================================================
    # AI 走子
    # =========================================================================

    def _on_ai_move(self) -> None:
        """在后台线程执行 AI 搜索。"""
        if self._ai_thinking:
            return
        if Rules.is_game_over(self._board):
            self._on_game_over()
            return

        current: int = self._board.get_current_player()
        player_name: str = "白方" if current == WHITE else "黑方"

        self._ai_thinking = True
        self._update_status(f"AI（{player_name}）思考中...")

        # 在后台线程搜索
        def search_and_update() -> None:
            time_budget: float = 15000.0  # 15秒
            move: Move = self._engine.search(self._board.clone(), time_budget)
            self._root.after(0, lambda: self._apply_ai_move(move, current))

        thread: threading.Thread = threading.Thread(target=search_and_update, daemon=True)
        thread.start()

    def _apply_ai_move(self, move: Move, player: int) -> None:
        """在主线程中应用 AI 走法。

        Args:
            move: AI 选择的走法
            player: 走子方（WHITE/BLACK）
        """
        self._board.make_move(move)
        self._board_canvas.set_last_move(move)
        self._ai_thinking = False

        # 记录走法
        player_type: str = "AI白" if player == WHITE else "AI黑"
        self._move_count += 1
        self._move_log.insert(tk.END, f"{self._move_count:>2d}. [{player_type}] {move.to_string()}")
        self._move_log.see(tk.END)  # 自动滚动到最新

        self._update_display()

        if Rules.is_game_over(self._board):
            self._on_game_over()
        elif self._is_both_ai():
            # AI vs AI 继续
            self._root.after(500, self._on_ai_move)
        else:
            # AI 走完，轮到人类 → 持续提示
            self._update_status("AI已操作——请走子")

    # =========================================================================
    # 人类走子
    # =========================================================================

    def _execute_human_move(self, move: Move) -> None:
        """执行人类走法。

        Args:
            move: 人类输入的走法
        """
        current: int = self._board.get_current_player()
        self._board.make_move(move)
        self._board_canvas.set_last_move(move)

        # 记录走法
        player_type: str = "人类白" if current == WHITE else "人类黑"
        self._move_count += 1
        self._move_log.insert(tk.END, f"{self._move_count:>2d}. [{player_type}] {move.to_string()}")
        self._move_log.see(tk.END)

        self._update_display()

        if Rules.is_game_over(self._board):
            self._on_game_over()
        elif not self._is_both_ai():
            # 人类走完 → 轮到 AI（如果对方是 AI）
            next_player: int = self._board.get_current_player()
            if not self._is_human_player(next_player):
                self._root.after(300, self._on_ai_move)

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _is_human_player(self, player: int) -> bool:
        """判断某方是否为人类操控。"""
        if player == WHITE:
            return self._white_player == "human"
        return self._black_player == "human"

    def _is_both_ai(self) -> bool:
        """判断是否双方都是 AI。"""
        return self._white_player == "ai" and self._black_player == "ai"

    def _reset_input(self) -> None:
        """重置输入状态机。"""
        self._input_state = self.IDLE
        self._selected_amazon = None
        self._selected_move_target = None

    def _update_display(self) -> None:
        """更新棋盘显示和状态栏。"""
        self._board_canvas.draw_board(self._board)
        if self._ai_thinking:
            return  # 状态由 _on_ai_move 设置
        # 非 AI 思考中 → 根据当前玩家设置状态
        current: int = self._board.get_current_player()
        if self._is_human_player(current):
            player_name: str = "白方" if current == WHITE else "黑方"
            self._update_status(f"当前: {player_name}（你的回合）")

    def _update_status(self, text: str) -> None:
        """更新状态栏文本。"""
        self._status_label.config(text=text)

    def _is_legal_move_target(self, row: int, col: int) -> bool:
        """检查 (row, col) 是否是从 selected_amazon 的合法移动目标。"""
        if self._selected_amazon is None:
            return False
        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(
            self._board, self._selected_amazon
        )
        return (row, col) in reachable

    def _is_legal_arrow_target(self, row: int, col: int) -> bool:
        """检查 (row, col) 是否是从 selected_move_target 的合法箭落点。"""
        if self._selected_move_target is None:
            return False
        reachable: list[tuple[int, int]] = Rules.get_queen_reachable(
            self._board, self._selected_move_target
        )
        return (row, col) in reachable

    def _on_new_game(self) -> None:
        """新局按钮回调——弹出模式选择后重启。"""
        self._show_mode_dialog_and_start()

    def _on_undo(self) -> None:
        """悔棋按钮回调——撤销两步（AI+人类各一）。"""
        if self._ai_thinking:
            return
        try:
            self._board.undo_move()
            if not self._is_both_ai():
                self._board.undo_move()
            # 撤销走法记录
            remove_count: int = 2 if not self._is_both_ai() else 1
            for _ in range(remove_count):
                if self._move_log.size() > 0:
                    self._move_log.delete(tk.END)
                    self._move_count -= 1
        except IndexError:
            show_error_dialog(self._root, "无法悔棋（走法历史不足）")
            return
        self._reset_input()
        self._update_display()

    def _on_game_over(self) -> None:
        """终局处理。"""
        winner: int = Rules.get_winner(self._board)
        winner_name: str = "白方" if winner == WHITE else "黑方"
        self._update_status(f"游戏结束——{winner_name} 获胜！")
        show_game_over_dialog(self._root, winner_name)
