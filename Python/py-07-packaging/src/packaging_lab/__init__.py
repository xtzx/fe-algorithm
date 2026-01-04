"""
Packaging Lab - Python 包管理学习示例

这是一个用于学习 Python 包管理的示例项目。
"""

__version__ = "0.1.0"
__author__ = "Developer"

from .core import greet, calculate

__all__ = ["greet", "calculate", "__version__"]

