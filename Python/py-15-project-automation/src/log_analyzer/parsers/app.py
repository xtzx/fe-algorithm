"""
应用日志解析器

支持的格式：
- 标准格式：2024-01-01 12:00:00 [LEVEL] message
- Python logging 格式：2024-01-01 12:00:00,123 - logger - LEVEL - message
"""

import re
from datetime import datetime

from log_analyzer.models import AppLogEntry, LogLevel
from log_analyzer.parsers.base import BaseParser

# 标准格式
# 2024-01-01 12:00:00 [ERROR] Something went wrong
STANDARD_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.:]\d+)?)\s+"
    r"\[?(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\]?\s+"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)

# Python logging 格式
# 2024-01-01 12:00:00,123 - myapp.module - ERROR - message
PYTHON_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.:]\d+)?)\s+"
    r"-\s+(?P<logger>\S+)\s+"
    r"-\s+(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+"
    r"-\s+(?P<message>.*)$",
    re.IGNORECASE,
)


def parse_level(level_str: str) -> LogLevel:
    """解析日志级别"""
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "WARN": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
        "FATAL": LogLevel.CRITICAL,
    }
    return level_map.get(level_str.upper(), LogLevel.INFO)


def parse_timestamp(ts_str: str) -> datetime:
    """解析时间戳"""
    # 标准化分隔符
    ts_str = ts_str.replace(",", ".").replace(":", ".", 3)

    formats = [
        "%Y-%m-%d %H.%M.%S.%f",
        "%Y-%m-%d %H.%M.%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue

    return datetime.now()


class AppLogParser(BaseParser):
    """应用日志解析器"""

    def parse_line(self, line: str, line_number: int = 0) -> AppLogEntry | None:
        # 尝试 Python logging 格式
        match = PYTHON_PATTERN.match(line)
        if match:
            groups = match.groupdict()
            return AppLogEntry(
                timestamp=parse_timestamp(groups["timestamp"]),
                level=parse_level(groups["level"]),
                message=groups["message"],
                logger_name=groups.get("logger", ""),
            )

        # 尝试标准格式
        match = STANDARD_PATTERN.match(line)
        if match:
            groups = match.groupdict()
            return AppLogEntry(
                timestamp=parse_timestamp(groups["timestamp"]),
                level=parse_level(groups["level"]),
                message=groups["message"],
            )

        return None

