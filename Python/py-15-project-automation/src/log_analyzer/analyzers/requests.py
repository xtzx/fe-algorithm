"""
请求分析器

分析 Nginx 访问日志中的请求统计
"""

from collections import Counter

from log_analyzer.models import LogEntry, NginxLogEntry, RequestStats


class RequestAnalyzer:
    """请求分析器"""

    def __init__(self) -> None:
        self.total_requests = 0
        self.status_counter: Counter[int] = Counter()
        self.method_counter: Counter[str] = Counter()
        self.url_counter: Counter[str] = Counter()
        self.response_times: list[float] = []
        self.error_count = 0

    def analyze_entry(self, entry: LogEntry) -> None:
        """分析单条日志"""
        if not isinstance(entry, NginxLogEntry):
            return

        self.total_requests += 1
        self.status_counter[entry.status_code] += 1
        self.method_counter[entry.request_method] += 1
        self.url_counter[entry.request_uri] += 1

        if entry.response_time > 0:
            self.response_times.append(entry.response_time)

        if entry.status_code >= 500:
            self.error_count += 1

    def analyze_entries(self, entries: list[LogEntry]) -> None:
        """分析多条日志"""
        for entry in entries:
            self.analyze_entry(entry)

    def get_stats(self) -> RequestStats:
        """获取统计结果"""
        avg_time = 0.0
        max_time = 0.0
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            max_time = max(self.response_times)

        error_rate = 0.0
        if self.total_requests > 0:
            error_rate = self.error_count / self.total_requests * 100

        return RequestStats(
            total_requests=self.total_requests,
            by_status_code=dict(self.status_counter),
            by_method=dict(self.method_counter),
            top_urls=self.url_counter.most_common(10),
            avg_response_time=round(avg_time, 3),
            max_response_time=round(max_time, 3),
            error_rate=round(error_rate, 2),
        )

    def reset(self) -> None:
        """重置分析器"""
        self.total_requests = 0
        self.status_counter.clear()
        self.method_counter.clear()
        self.url_counter.clear()
        self.response_times.clear()
        self.error_count = 0

