"""
错误分析器

分析日志中的错误和警告
"""

from collections import Counter

from log_analyzer.models import ErrorStats, LogEntry, LogLevel


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self) -> None:
        self.level_counter: Counter[str] = Counter()
        self.hour_counter: Counter[int] = Counter()
        self.message_counter: Counter[str] = Counter()
        self.total_errors = 0
        self.total_warnings = 0
        self.total_critical = 0

    def analyze_entry(self, entry: LogEntry) -> None:
        """分析单条日志"""
        if entry.level in (LogLevel.ERROR, LogLevel.WARNING, LogLevel.CRITICAL):
            self.level_counter[entry.level.value] += 1
            self.hour_counter[entry.timestamp.hour] += 1

            # 截取消息前100字符作为 key
            msg_key = entry.message[:100] if entry.message else "(empty)"
            self.message_counter[msg_key] += 1

            if entry.level == LogLevel.ERROR:
                self.total_errors += 1
            elif entry.level == LogLevel.WARNING:
                self.total_warnings += 1
            elif entry.level == LogLevel.CRITICAL:
                self.total_critical += 1

    def analyze_entries(self, entries: list[LogEntry]) -> None:
        """分析多条日志"""
        for entry in entries:
            self.analyze_entry(entry)

    def get_stats(self) -> ErrorStats:
        """获取统计结果"""
        return ErrorStats(
            total_errors=self.total_errors,
            total_warnings=self.total_warnings,
            total_critical=self.total_critical,
            by_level=dict(self.level_counter),
            by_hour=dict(self.hour_counter),
            top_messages=self.message_counter.most_common(10),
        )

    def reset(self) -> None:
        """重置分析器"""
        self.level_counter.clear()
        self.hour_counter.clear()
        self.message_counter.clear()
        self.total_errors = 0
        self.total_warnings = 0
        self.total_critical = 0

