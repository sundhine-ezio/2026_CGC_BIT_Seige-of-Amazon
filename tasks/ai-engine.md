# ai-engine 任务清单

> **模块**：`src/ai/engine.py`  
> **依赖**：`game-board`, `game-move`  
> **预估工时**：0.5 小时  

## 实现任务

- [ ] **E-01** 创建 `src/ai/__init__.py`（空文件）
- [ ] **E-02** 创建 `src/ai/engine.py`，导入 `ABC`, `abstractmethod`, `Board`, `Move`

- [ ] **E-03** 定义 `Engine` 抽象基类
  ```python
  class Engine(ABC):
      @abstractmethod
      def search(self, board: Board, time_budget_ms: float) -> Move:
          """搜索并返回最优走法"""
          ...

      @abstractmethod
      def get_name(self) -> str:
          """返回引擎名称"""
          ...

      @abstractmethod
      def reset(self) -> None:
          """重置引擎内部状态"""
          ...
  ```

- [ ] **E-04** 验证抽象基类正确性：尝试实例化 `Engine()` 应抛 `TypeError`

## 测试任务

- [ ] **T-01** 创建 `tests/test_engine.py`
- [ ] **T-02** 测试实例化抽象类抛 `TypeError`
- [ ] **T-03** 测试子类未实现 `search` 方法抛 `TypeError`
- [ ] **T-04** 测试正确实现的子类可实例化
- [ ] **T-05** 运行 `python -m pytest tests/test_engine.py -v`，全部通过

## 完成标准
- [ ] `Engine` 为合法 ABC，定义三个抽象方法签名
- [ ] 子类必须实现全部抽象方法才能实例化
- [ ] 全部测试通过
