"""
JSON 日志解析器

支持结构化 JSON 日志
"""

import json
from datetime import datetime
from typing import Any

from log_analyzer.models import JsonLogEntry, LogLevel
from log_analyzer.parsers.base import BaseParser


def parse_level(data: dict[str, Any]) -> LogLevel:
    """从 JSON 中解析日志级别"""
    level_keys = ["level", "severity", "log_level", "loglevel"]

    for key in level_keys:
        if key in data:
            level_str = str(data[key]).upper()
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "WARN": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL,
                "FATAL": LogLevel.CRITICAL,
            }
            return level_map.get(level_str, LogLevel.INFO)

    return LogLevel.INFO


def parse_timestamp(data: dict[str, Any]) -> datetime:
    """从 JSON 中解析时间戳"""
    ts_keys = ["timestamp", "time", "@timestamp", "datetime", "ts", "date"]

    for key in ts_keys:
        if key in data:
            ts_value = data[key]

            # 已经是数字（Unix 时间戳）
            if isinstance(ts_value, (int, float)):
                return datetime.fromtimestamp(ts_value)

            # 字符串格式
            if isinstance(ts_value, str):
                formats = [
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(ts_value, fmt)
                    except ValueError:
                        continue

    return datetime.now()


def parse_message(data: dict[str, Any]) -> str:
    """从 JSON 中解析消息"""
    msg_keys = ["message", "msg", "text", "log", "body"]

    for key in msg_keys:
        if key in data:
            return str(data[key])

    return ""


class JsonLogParser(BaseParser):
    """JSON 日志解析器"""

    def parse_line(self, line: str, line_number: int = 0) -> JsonLogEntry | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # 提取标准字段后，剩余的放入 extra
        standard_keys = {
            "timestamp",
            "time",
            "@timestamp",
            "datetime",
            "ts",
            "date",
            "level",
            "severity",
            "log_level",
            "loglevel",
            "message",
            "msg",
            "text",
            "log",
            "body",
        }

        extra = {k: v for k, v in data.items() if k not in standard_keys}

        return JsonLogEntry(
            timestamp=parse_timestamp(data),
            level=parse_level(data),
            message=parse_message(data),
            extra=extra,
        )

