"""
数据模型

使用 pydantic 定义日志和统计数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """日志级别"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    """日志条目基类"""

    timestamp: datetime
    level: LogLevel = LogLevel.INFO
    message: str = ""
    source_file: str = ""
    line_number: int = 0
    raw_line: str = ""

    def is_error(self) -> bool:
        return self.level in (LogLevel.ERROR, LogLevel.CRITICAL)

    def is_warning(self) -> bool:
        return self.level == LogLevel.WARNING


class NginxLogEntry(LogEntry):
    """Nginx 访问日志条目"""

    remote_addr: str = ""
    request_method: str = ""
    request_uri: str = ""
    status_code: int = 0
    body_bytes_sent: int = 0
    response_time: float = 0.0
    http_referer: str = ""
    http_user_agent: str = ""

    def is_error(self) -> bool:
        return self.status_code >= 500

    def is_client_error(self) -> bool:
        return 400 <= self.status_code < 500


class AppLogEntry(LogEntry):
    """应用日志条目"""

    logger_name: str = ""
    module: str = ""
    function: str = ""
    exception: str | None = None


class JsonLogEntry(LogEntry):
    """JSON 格式日志条目"""

    extra: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 统计模型
# ============================================================================


class ErrorStats(BaseModel):
    """错误统计"""

    total_errors: int = 0
    total_warnings: int = 0
    total_critical: int = 0
    by_level: dict[str, int] = Field(default_factory=dict)
    by_hour: dict[int, int] = Field(default_factory=dict)
    top_messages: list[tuple[str, int]] = Field(default_factory=list)


class RequestStats(BaseModel):
    """请求统计"""

    total_requests: int = 0
    by_status_code: dict[int, int] = Field(default_factory=dict)
    by_method: dict[str, int] = Field(default_factory=dict)
    top_urls: list[tuple[str, int]] = Field(default_factory=list)
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    error_rate: float = 0.0


class TimelineStats(BaseModel):
    """时间分布统计"""

    by_hour: dict[int, int] = Field(default_factory=dict)
    by_day: dict[str, int] = Field(default_factory=dict)
    peak_hour: int = 0
    peak_count: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None


class AnalysisReport(BaseModel):
    """分析报告"""

    generated_at: datetime = Field(default_factory=datetime.now)
    files_analyzed: int = 0
    total_entries: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    error_stats: ErrorStats = Field(default_factory=ErrorStats)
    request_stats: RequestStats | None = None
    timeline_stats: TimelineStats = Field(default_factory=TimelineStats)


# ============================================================================
# 清理相关模型
# ============================================================================


class CleanupAction(str, Enum):
    """清理动作"""

    ARCHIVE = "archive"
    COMPRESS = "compress"
    DELETE = "delete"


class CleanupTask(BaseModel):
    """清理任务"""

    source_path: str
    action: CleanupAction
    target_path: str | None = None
    size_bytes: int = 0
    modified_time: datetime | None = None


class CleanupState(BaseModel):
    """清理状态（用于断点续跑）"""

    batch_id: str
    created_at: datetime
    total_tasks: int
    completed: list[int] = Field(default_factory=list)
    failed: list[int] = Field(default_factory=list)
    pending: list[int] = Field(default_factory=list)


class CleanupResult(BaseModel):
    """清理结果"""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    bytes_freed: int = 0
    bytes_archived: int = 0
    errors: list[str] = Field(default_factory=list)

