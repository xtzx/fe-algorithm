"""
Python 项目工程化脚手架

这是一个可复用的项目模板，包含：
- 配置管理 (pydantic-settings)
- 日志配置
- CLI 框架
- 工具链集成
"""

from scaffold.config import Settings, get_settings
from scaffold.log import get_logger, setup_logging

__version__ = "0.1.0"

__all__ = [
    # 配置
    "Settings",
    "get_settings",
    # 日志
    "setup_logging",
    "get_logger",
    # 版本
    "__version__",
]

