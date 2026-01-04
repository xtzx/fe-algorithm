"""日志解析器模块"""

from log_analyzer.parsers.app import AppLogParser
from log_analyzer.parsers.base import BaseParser, ParseError
from log_analyzer.parsers.json_log import JsonLogParser
from log_analyzer.parsers.nginx import NginxLogParser

__all__ = [
    "BaseParser",
    "ParseError",
    "NginxLogParser",
    "AppLogParser",
    "JsonLogParser",
]


def get_parser(format_type: str) -> BaseParser:
    """根据格式类型获取解析器"""
    parsers = {
        "nginx": NginxLogParser,
        "app": AppLogParser,
        "json": JsonLogParser,
    }

    if format_type not in parsers:
        raise ValueError(f"Unknown log format: {format_type}")

    return parsers[format_type]()


def detect_format(line: str) -> str | None:
    """自动检测日志格式"""
    line = line.strip()

    # JSON 格式
    if line.startswith("{") and line.endswith("}"):
        return "json"

    # Nginx 格式（IP 开头）
    import re

    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", line):
        return "nginx"

    # App 格式（时间戳开头）
    if re.match(r"^\d{4}-\d{2}-\d{2}", line):
        return "app"

    return None

