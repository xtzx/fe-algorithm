"""测试配置和共享 fixture"""

from datetime import datetime
from pathlib import Path

import pytest

from log_analyzer.models import AppLogEntry, JsonLogEntry, LogLevel, NginxLogEntry


@pytest.fixture
def sample_nginx_lines() -> list[str]:
    """示例 Nginx 日志行"""
    return [
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0800] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"',
        '192.168.1.2 - - [01/Jan/2024:12:01:00 +0800] "POST /api/login HTTP/1.1" 401 56 "-" "curl/7.68.0"',
        '192.168.1.3 - - [01/Jan/2024:12:02:00 +0800] "GET /api/products HTTP/1.1" 500 0 "-" "Mozilla/5.0"',
    ]


@pytest.fixture
def sample_app_lines() -> list[str]:
    """示例应用日志行"""
    return [
        "2024-01-01 12:00:00 [INFO] Application started",
        "2024-01-01 12:01:00 [WARNING] Memory usage high: 80%",
        "2024-01-01 12:02:00 [ERROR] Database connection failed",
        "2024-01-01 12:03:00 [CRITICAL] Service unavailable",
    ]


@pytest.fixture
def sample_json_lines() -> list[str]:
    """示例 JSON 日志行"""
    return [
        '{"timestamp": "2024-01-01T12:00:00", "level": "INFO", "message": "User logged in", "user_id": 123}',
        '{"timestamp": "2024-01-01T12:01:00", "level": "ERROR", "message": "API timeout", "endpoint": "/api/data"}',
        '{"timestamp": "2024-01-01T12:02:00", "level": "WARNING", "message": "Rate limit exceeded", "ip": "192.168.1.1"}',
    ]


@pytest.fixture
def sample_nginx_entries() -> list[NginxLogEntry]:
    """示例 Nginx 日志条目"""
    return [
        NginxLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="GET /api/users 200",
            remote_addr="192.168.1.1",
            request_method="GET",
            request_uri="/api/users",
            status_code=200,
            body_bytes_sent=1234,
            response_time=0.05,
        ),
        NginxLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
            level=LogLevel.WARNING,
            message="POST /api/login 401",
            remote_addr="192.168.1.2",
            request_method="POST",
            request_uri="/api/login",
            status_code=401,
            body_bytes_sent=56,
            response_time=0.02,
        ),
        NginxLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 2, 0),
            level=LogLevel.ERROR,
            message="GET /api/products 500",
            remote_addr="192.168.1.3",
            request_method="GET",
            request_uri="/api/products",
            status_code=500,
            body_bytes_sent=0,
            response_time=1.5,
        ),
    ]


@pytest.fixture
def sample_app_entries() -> list[AppLogEntry]:
    """示例应用日志条目"""
    return [
        AppLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Application started",
        ),
        AppLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
            level=LogLevel.WARNING,
            message="Memory usage high: 80%",
        ),
        AppLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 2, 0),
            level=LogLevel.ERROR,
            message="Database connection failed",
        ),
        AppLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 3, 0),
            level=LogLevel.CRITICAL,
            message="Service unavailable",
        ),
    ]


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """创建临时日志目录"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # 创建示例日志文件
    nginx_log = log_dir / "nginx.log"
    nginx_log.write_text(
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0800] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"\n'
        '192.168.1.2 - - [01/Jan/2024:12:01:00 +0800] "GET /api/products HTTP/1.1" 500 0 "-" "Mozilla/5.0"\n'
    )

    app_log = log_dir / "app.log"
    app_log.write_text(
        "2024-01-01 12:00:00 [INFO] Application started\n"
        "2024-01-01 12:01:00 [ERROR] Database error\n"
    )

    return log_dir

