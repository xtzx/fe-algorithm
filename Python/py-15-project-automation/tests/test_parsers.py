"""解析器测试"""

import pytest

from log_analyzer.models import LogLevel
from log_analyzer.parsers import (
    AppLogParser,
    JsonLogParser,
    NginxLogParser,
    detect_format,
)


class TestNginxLogParser:
    """Nginx 日志解析器测试"""

    def test_parse_combined_format(self, sample_nginx_lines: list[str]) -> None:
        parser = NginxLogParser()

        entry = parser.parse_line(sample_nginx_lines[0])
        assert entry is not None
        assert entry.remote_addr == "192.168.1.1"
        assert entry.request_method == "GET"
        assert entry.request_uri == "/api/users"
        assert entry.status_code == 200
        assert entry.level == LogLevel.INFO

    def test_parse_error_status(self, sample_nginx_lines: list[str]) -> None:
        parser = NginxLogParser()

        entry = parser.parse_line(sample_nginx_lines[2])
        assert entry is not None
        assert entry.status_code == 500
        assert entry.level == LogLevel.ERROR
        assert entry.is_error() is True

    def test_parse_client_error_status(self, sample_nginx_lines: list[str]) -> None:
        parser = NginxLogParser()

        entry = parser.parse_line(sample_nginx_lines[1])
        assert entry is not None
        assert entry.status_code == 401
        assert entry.level == LogLevel.WARNING

    def test_parse_invalid_line(self) -> None:
        parser = NginxLogParser()

        entry = parser.parse_line("invalid log line")
        assert entry is None


class TestAppLogParser:
    """应用日志解析器测试"""

    def test_parse_standard_format(self, sample_app_lines: list[str]) -> None:
        parser = AppLogParser()

        entry = parser.parse_line(sample_app_lines[0])
        assert entry is not None
        assert entry.level == LogLevel.INFO
        assert entry.message == "Application started"

    def test_parse_all_levels(self, sample_app_lines: list[str]) -> None:
        parser = AppLogParser()

        levels = [LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]

        for line, expected_level in zip(sample_app_lines, levels):
            entry = parser.parse_line(line)
            assert entry is not None
            assert entry.level == expected_level

    def test_parse_python_logging_format(self) -> None:
        parser = AppLogParser()

        line = "2024-01-01 12:00:00,123 - myapp.module - ERROR - Something failed"
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.level == LogLevel.ERROR
        assert entry.message == "Something failed"
        assert entry.logger_name == "myapp.module"


class TestJsonLogParser:
    """JSON 日志解析器测试"""

    def test_parse_json_log(self, sample_json_lines: list[str]) -> None:
        parser = JsonLogParser()

        entry = parser.parse_line(sample_json_lines[0])
        assert entry is not None
        assert entry.level == LogLevel.INFO
        assert entry.message == "User logged in"
        assert entry.extra.get("user_id") == 123

    def test_parse_extra_fields(self, sample_json_lines: list[str]) -> None:
        parser = JsonLogParser()

        entry = parser.parse_line(sample_json_lines[1])
        assert entry is not None
        assert entry.extra.get("endpoint") == "/api/data"

    def test_parse_invalid_json(self) -> None:
        parser = JsonLogParser()

        entry = parser.parse_line("not valid json")
        assert entry is None


class TestDetectFormat:
    """格式自动检测测试"""

    def test_detect_nginx(self) -> None:
        line = '192.168.1.1 - - [01/Jan/2024:12:00:00 +0800] "GET / HTTP/1.1" 200 1234'
        assert detect_format(line) == "nginx"

    def test_detect_app(self) -> None:
        line = "2024-01-01 12:00:00 [INFO] message"
        assert detect_format(line) == "app"

    def test_detect_json(self) -> None:
        line = '{"timestamp": "2024-01-01", "level": "INFO"}'
        assert detect_format(line) == "json"

    def test_detect_unknown(self) -> None:
        line = "some random text"
        assert detect_format(line) is None

