"""GUI 对话框工具函数。"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


def show_game_over_dialog(parent: tk.Tk, winner_name: str) -> None:
    """游戏结束弹窗。

    Args:
        parent: 父窗口
        winner_name: 胜者名称（如 "白方"、"黑方"）
    """
    messagebox.showinfo(
        "游戏结束",
        f"{winner_name} 获胜！",
        parent=parent,
    )


def show_mode_selection_dialog(parent: tk.Tk) -> dict[str, str] | None:
    """对局模式选择对话框——分别选择白方和黑方的操作者。

    Args:
        parent: 父窗口

    Returns:
        {"white": "human"|"ai", "black": "human"|"ai"} 或 None（取消）
    """
    dialog: tk.Toplevel = tk.Toplevel(parent)
    dialog.title("选择对局模式")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    # 居中
    dialog.geometry("300x180")
    dialog.update_idletasks()
    x: int = parent.winfo_rootx() + (parent.winfo_width() - 300) // 2
    y: int = parent.winfo_rooty() + (parent.winfo_height() - 180) // 2
    dialog.geometry(f"+{x}+{y}")

    result: dict[str, str] = {}

    # ---- 白方 ----
    tk.Label(dialog, text="白方 ●", font=("Arial", 12, "bold")).grid(
        row=0, column=0, padx=15, pady=(15, 5), sticky="w"
    )
    white_var: tk.StringVar = tk.StringVar(value="human")
    tk.Radiobutton(dialog, text="人类", variable=white_var, value="human",
                   font=("Arial", 11)).grid(row=0, column=1, padx=5, sticky="w")
    tk.Radiobutton(dialog, text="AI  ", variable=white_var, value="ai",
                   font=("Arial", 11)).grid(row=0, column=2, padx=5, sticky="w")

    # ---- 黑方 ----
    tk.Label(dialog, text="黑方 ○", font=("Arial", 12, "bold")).grid(
        row=1, column=0, padx=15, pady=(10, 5), sticky="w"
    )
    black_var: tk.StringVar = tk.StringVar(value="ai")
    tk.Radiobutton(dialog, text="人类", variable=black_var, value="human",
                   font=("Arial", 11)).grid(row=1, column=1, padx=5, sticky="w")
    tk.Radiobutton(dialog, text="AI  ", variable=black_var, value="ai",
                   font=("Arial", 11)).grid(row=1, column=2, padx=5, sticky="w")

    # ---- 按钮 ----
    def _on_ok() -> None:
        result["white"] = white_var.get()
        result["black"] = black_var.get()
        dialog.destroy()

    def _on_cancel() -> None:
        dialog.destroy()

    btn_frame: tk.Frame = tk.Frame(dialog)
    btn_frame.grid(row=2, column=0, columnspan=3, pady=20)

    tk.Button(btn_frame, text="确定", command=_on_ok, width=10).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btn_frame, text="取消", command=_on_cancel, width=10).pack(
        side=tk.LEFT, padx=10
    )

    # 等待对话框关闭
    parent.wait_window(dialog)

    if not result:
        return None
    return result


def show_error_dialog(parent: tk.Tk, message: str) -> None:
    """错误提示弹窗。

    Args:
        parent: 父窗口
        message: 错误信息
    """
    messagebox.showerror("错误", message, parent=parent)
