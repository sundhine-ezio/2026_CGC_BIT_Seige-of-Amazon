# main-and-config 任务清单

> **模块**：`main.py` + `config.yaml`  
> **依赖**：`ai-alphabeta`, `opening-book`, `endgame`, `gui`  
> **预估工时**：2 小时  

## 实现任务

### config.yaml

- [ ] **MC-01** 创建项目根目录 `config.yaml`
- [ ] **MC-02** 写入完整配置项（按详细设计文档第 10 节模板）：
  - `game`: board_size, num_amazons
  - `alphabeta`: max_depth, time_budget_ms, use_transposition_table, use_iterative_deepening
  - `evaluator`: type, num_simulations
  - `mcts`: (占位)
  - `opening_book`: enabled, book_path
  - `endgame`: enabled, tablebase_path, max_pieces, max_empty_squares
  - `gui`: cell_size, piece_radius, board_color, default_mode
  - `logging`: level, game_log_enabled, game_log_path

### 配置文件加载

- [ ] **MC-03** 创建 `src/config_loader.py`
- [ ] **MC-04** 实现 `load_config(path="config.yaml") → dict`
  - 用 `yaml.safe_load()` 解析（需要 `pip install pyyaml`）
  - 返回嵌套字典

### main.py

- [ ] **MC-05** 创建项目根目录 `main.py`

- [ ] **MC-06** 实现 `parse_args() → argparse.Namespace`
  ```python
  --mode        {gui, console, competition}  (默认: gui)
  --color       {white, black}              (比赛模式用)
  --time        int                          (比赛模式总时间，秒)
  --config      str                          (配置文件路径，默认: config.yaml)
  --depth       int                          (覆盖搜索深度)
  --book        str                          (开局库路径)
  --tablebase   str                          (残局库路径)
  ```

- [ ] **MC-07** 实现 `setup_engine(config, args) → Engine`
  1. 创建 `MonteCarloEvaluator(num_simulations=...)`
  2. 创建 `AlphaBetaEngine(evaluator, ...)`
  3. 若配置启用，创建 `OpeningBook` 并注入
  4. 若配置启用，创建 `Tablebase` 并注入
  5. 返回 engine

- [ ] **MC-08** 实现 `run_gui_mode(engine, args)`
  - 导入 `AmazonsGUI`
  - `gui = AmazonsGUI(engine, mode=config["gui"]["default_mode"])`
  - `gui.run()`

- [ ] **MC-09** 实现 `run_console_mode(engine_white, engine_black, args)`
  1. 初始化 `Board`
  2. 循环至终局：
     - 文本打印当前棋盘（简单的 10×10 字符矩阵）
     - 调用当前方 engine.search() 获取走法
     - 打印走法字符串
     - `board.make_move(move)`
  3. 打印胜者
  4. 输出对局统计（总步数、搜索耗时）

- [ ] **MC-10** 实现 `run_competition_mode(engine, args)`
  1. 初始化 `Board`
  2. 打印 `"READY"` 到 stdout
  3. 若己方执白：搜索走法 → 打印走法字符串 → `board.make_move()`
  4. 循环：
     - 从 `sys.stdin.readline()` 读取对手走法（去除换行符）
     - 若读到 `"RESIGN"` 或空行 → 退出
     - `Move.from_string()` 解析 → `board.make_move()`
     - 若终局 → 打印 `"RESIGN"` → 退出
     - `engine.search()` 搜索 → 打印走法字符串 → `board.make_move()`
     - 若终局 → 退出
  5. 所有输出用 `sys.stdout.flush()` 确保即时刷新

- [ ] **MC-11** 实现 `main()` 入口
  ```python
  if __name__ == "__main__":
      args = parse_args()
      config = load_config(args.config)
      engine = setup_engine(config, args)
      if args.mode == "gui":
          run_gui_mode(engine, args)
      elif args.mode == "console":
          run_console_mode(engine, engine, args)
      elif args.mode == "competition":
          run_competition_mode(engine, args)
  ```

- [ ] **MC-12** 创建 `requirements.txt`
  ```
  numpy>=1.24
  pyyaml>=6.0
  torch>=2.0  (第二阶段)
  pytest>=7.0
  ```

## 测试任务

- [ ] **T-01** 测试 `load_config()` 返回正确的嵌套字典
- [ ] **T-02** 测试 `parse_args` 默认值正确
- [ ] **T-03** 冒烟测试：`python main.py --mode console --ai-white --ai-black --depth 1`
  - 程序启动不崩溃
  - 输出至少一步合法走法
- [ ] **T-04** 比赛模式模拟测试
  ```bash
  echo "d1-d5-g8" | python main.py --mode competition --color black --depth 2
  ```
  - 程序输出 "READY"
  - 收到对手走法后输出己方走法
- [ ] **T-05** 完整对弈：`python main.py --mode console --depth 3`
  - 程序从初始状态完成一整局
  - 输出胜者

## 完成标准
- [ ] `config.yaml` 加载解析正确
- [ ] 三种模式（gui / console / competition）均可正常启动
- [ ] 比赛模式 stdin/stdout 通信正确
- [ ] `requirements.txt` 包含全部依赖
