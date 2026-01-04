"""Code Counter - 代码统计工具

一个功能完整的命令行代码统计工具，支持多种编程语言的行数统计。

使用方式:
    code-counter scan <path>
    code-counter report <path>
    code-counter config show
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from code_counter.models import FileStats, LanguageStats, ScanResult
from code_counter.scanner import Scanner
from code_counter.counter import Counter
from code_counter.config import Config

__all__ = [
    "FileStats",
    "LanguageStats",
    "ScanResult",
    "Scanner",
    "Counter",
    "Config",
]

