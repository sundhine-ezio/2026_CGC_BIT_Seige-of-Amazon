# opening-book 任务清单

> **模块**：`src/opening_book/book.py` + `data/opening_book.json`  
> **依赖**：`game-move`, `game-board`  
> **预估工时**：1.5 小时  

## 实现任务

### 数据文件

- [ ] **OB-01** 创建 `src/opening_book/__init__.py`（空文件）
- [ ] **OB-02** 创建 `src/opening_book/data/` 目录
- [ ] **OB-03** 创建 `src/opening_book/data/opening_book.json`，包含至少 2-3 条手工录入的开局走法
  ```json
  {
    "version": "1.0",
    "description": "Amazons opening book",
    "positions": {
      "<initial_hash_hex>": [
        {"move": "d1-d5-g8", "weight": 100, "comment": "中心突破"},
        {"move": "g1-d4-d8", "weight": 80, "comment": "侧翼发展"}
      ]
    }
  }
  ```
  > 初始哈希值可在完成 Board + Zobrist 后从初始棋盘的 `get_zobrist_hash()` 获取

### OpeningBook 类

- [ ] **OB-04** 创建 `src/opening_book/book.py`
- [ ] **OB-05** 实现 `OpeningBook.__init__(book_path=None)`
  - 若 `book_path` 为 None，跳过加载
  - 否则用 `json.load()` 加载文件
  - 将 `positions` 的键（十六进制字符串）转为 `int`
  - 加载失败抛 `FileNotFoundError` 或 `ValueError`

- [ ] **OB-06** 实现 `query(board_hash: int) → list[dict] | None`
  - 在内部字典中查找 `board_hash`
  - 命中返回候选列表，否则返回 None

- [ ] **OB-07** 实现 `get_random_move(board_hash: int) → Move | None`
  1. `query(board_hash)` 获取候选列表
  2. 若 None → 返回 None
  3. 提取 `weights = [item["weight"] for item in candidates]`
  4. 用 `random.choices(candidates, weights=weights, k=1)` 加权采样
  5. 调用 `Move.from_string(selected["move"])` 返回

- [ ] **OB-08** 实现 `is_available() → bool`

- [ ] **OB-09** 实现 `get_total_positions() → int`

## 测试任务

- [ ] **T-01** 创建 `tests/test_opening_book.py`
- [ ] **T-02** 准备测试用 JSON 文件（含 2 个已知哈希和走法）
- [ ] **T-03** 测试加载成功
- [ ] **T-04** 测试命中查询返回候选列表
- [ ] **T-05** 测试未命中查询返回 None
- [ ] **T-06** 测试 `get_random_move` 返回合法 Move
- [ ] **T-07** 测试加权随机：固定 seed，weight=0 的走法不被选中
- [ ] **T-08** 测试不存在文件 → `FileNotFoundError`
- [ ] **T-09** 测试 JSON 格式错误 → `ValueError`
- [ ] **T-10** 运行 `python -m pytest tests/test_opening_book.py -v`，全部通过

## 完成标准
- [ ] 开局库 JSON 正确加载
- [ ] 命中/未命中查询正确
- [ ] 加权随机工作正常
- [ ] 全部测试通过
