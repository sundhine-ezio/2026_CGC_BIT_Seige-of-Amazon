"""YAML 配置文件加载器。"""

from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        path: YAML 文件路径

    Returns:
        配置字典（嵌套结构）

    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML 解析错误
    """
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if config is None:
        config = {}

    return config
