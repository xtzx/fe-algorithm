"""
解析器基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from log_analyzer.models import LogEntry


class ParseError(Exception):
    """解析错误"""

    def __init__(self, message: str, line_number: int = 0, raw_line: str = ""):
        super().__init__(message)
        self.line_number = line_number
        self.raw_line = raw_line


class BaseParser(ABC):
    """日志解析器基类"""

    def __init__(self) -> None:
        self.errors: list[ParseError] = []

    @abstractmethod
    def parse_line(self, line: str, line_number: int = 0) -> LogEntry | None:
        """
        解析单行日志

        Args:
            line: 日志行
            line_number: 行号

        Returns:
            解析后的日志条目，失败返回 None
        """
        pass

    def parse_file(
        self,
        file_path: Path,
        *,
        skip_errors: bool = True,
    ) -> Iterator[LogEntry]:
        """
        解析日志文件

        Args:
            file_path: 日志文件路径
            skip_errors: 是否跳过解析错误

        Yields:
            日志条目
        """
        self.errors = []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = self.parse_line(line, line_number)
                    if entry:
                        entry.source_file = str(file_path)
                        entry.line_number = line_number
                        entry.raw_line = line
                        yield entry
                except Exception as e:
                    error = ParseError(str(e), line_number, line)
                    self.errors.append(error)
                    if not skip_errors:
                        raise error

    def parse_files(
        self,
        file_paths: list[Path],
        *,
        skip_errors: bool = True,
    ) -> Iterator[LogEntry]:
        """
        解析多个日志文件

        Args:
            file_paths: 日志文件路径列表
            skip_errors: 是否跳过解析错误

        Yields:
            日志条目
        """
        for file_path in file_paths:
            yield from self.parse_file(file_path, skip_errors=skip_errors)

