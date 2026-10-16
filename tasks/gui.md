# gui 任务清单

> **模块**：`src/gui/main_window.py` + `src/gui/board_painter.py` + `src/gui/dialogs.py`  
> **依赖**：`game-board`, `game-rules`, `ai-engine`  
> **预估工时**：3 小时  

## 实现任务

### board_painter.py

- [ ] **G-01** 创建 `src/gui/__init__.py`（空文件）
- [ ] **G-02** 创建 `src/gui/board_painter.py`

- [ ] **G-03** 定义视觉常量
  ```python
  CELL_SIZE = 60
  PIECE_RADIUS = 22
  ARROW_RADIUS = 5
  BOARD_PADDING = 30
  BOARD_COLOR = "#DEB887"
  LINE_COLOR = "#000000"
  ```

- [ ] **G-04** 实现 `BoardCanvas.__init__(parent)`
  - 继承 `tk.Canvas`
  - 画布尺寸 = `CELL_SIZE * 10 + 2 * PADDING`
  - 绑定 `<Button-1>` 点击事件

- [ ] **G-05** 实现 `set_click_callback(callback)`：注册点击回调 `callback(row, col)`

- [ ] **G-06** 实现 `_pixel_to_board(x, y) → tuple | None`
  - 像素坐标转棋盘行列
  - 点击在棋盘外返回 None

- [ ] **G-07** 实现 `_draw_piece(row, col, player)`
  - WHITE：白色圆形 + 黑色边框
  - BLACK：黑色圆形 + 白色边框

- [ ] **G-08** 实现 `_draw_burned(row, col)`
  - 灰色填充 + 红色 × 或暗色方块

- [ ] **G-09** 实现 `draw_board(board)`
  1. 清空画布
  2. 绘制 10×10 网格（底色 + 网格线）
  3. 绘制列标 a-j（顶部）、行标 1-10（右侧）
  4. 遍历 `board.get_grid()`，绘制棋子 / 烧毁格
  5. 高亮最近走法（如有）

- [ ] **G-10** 实现 `highlight_last_move(move)`
  - 从 from_pos 到 to_pos 画箭头线
  - arrow_pos 用红色标记

### main_window.py

- [ ] **G-11** 创建 `src/gui/main_window.py`

- [ ] **G-12** 实现 `AmazonsGUI.__init__(engine, mode="HUMAN_VS_AI")`
  - 继承 `tk.Tk`
  - 创建 `BoardCanvas`、状态栏 Label、按钮 Frame
  - 创建 `Board` 实例
  - 布局：棋盘上方、状态栏下方、按钮最下

- [ ] **G-13** 实现三阶段状态机 `_on_board_click(row, col)`
  ```python
  _input_state: str  # "IDLE" | "AMAZON_SELECTED" | "MOVE_SELECTED"
  _selected_amazon: tuple | None
  _selected_move_target: tuple | None
  ```
  - **IDLE**：点击己方亚马逊 → `AMAZON_SELECTED`（含高亮）
  - **AMAZON_SELECTED**：点击己方另一亚马逊 → 切换选择；点击合法移动目标 → `MOVE_SELECTED`
  - **MOVE_SELECTED**：点击合法箭落点 → 执行走法 → `IDLE`；否则忽略

- [ ] **G-14** 实现 `_is_legal_move_target(amazon, target) → bool`
  - 调用 `Rules.get_queen_reachable()` 验证

- [ ] **G-15** 实现 `_on_ai_move()`
  - 使用 `threading.Thread` 在后台运行 `engine.search()`
  - 搜索完成后用 `self.after(0, callback)` 回到主线程更新显示
  - 搜索期间在状态栏显示 "AI 思考中..."

- [ ] **G-16** 实现 `_update_display()`
  - `board_canvas.draw_board(board)`
  - 更新状态栏（当前玩家等）

- [ ] **G-17** 实现 `_on_new_game()`
  - `board.reset()`, `engine.reset()`
  - 重置输入状态机
  - `_update_display()`

- [ ] **G-18** 实现 `_on_undo()`
  - 人机模式：撤销两步（AI 一步 + 人类一步）
  - 更新显示

- [ ] **G-19** 实现 `run()`：调用 `self.mainloop()`

### dialogs.py

- [ ] **G-20** 创建 `src/gui/dialogs.py`
- [ ] **G-21** 实现 `show_game_over_dialog(parent, winner)`：`tk.messagebox` 显示胜者
- [ ] **G-22** 实现 `show_new_game_dialog(parent)`：选择模式的简单弹窗
- [ ] **G-23** 实现 `show_error_dialog(parent, message)`：错误提示

## 测试任务

- [ ] **T-01** 手工启动 GUI：`python main.py --mode gui`，窗口正常显示
- [ ] **T-02** 棋盘渲染：初始棋盘 8 枚棋子颜色和位置正确
- [ ] **T-03** 走法输入：点击己方亚马逊 → 点击移动目标 → 点击箭落点 → 走法执行
- [ ] **T-04** 非法点击：点击对方棋子/空格/烧毁格 → 被忽略
- [ ] **T-05** AI vs AI 观战模式：自动对弈，棋盘实时更新
- [ ] **T-06** 新局、悔棋按钮功能正常
- [ ] **T-07** 终局弹窗正常弹出

## 完成标准
- [ ] 棋盘渲染正确（棋子、烧毁格、坐标标注）
- [ ] 人类走法输入三阶段状态机正确
- [ ] AI 思考时不阻塞界面
- [ ] 手工走完一遍完整交互流程
