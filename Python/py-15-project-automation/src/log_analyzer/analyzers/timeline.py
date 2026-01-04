"""
时间分布分析器

分析日志的时间分布特征
"""

from collections import Counter
from datetime import datetime

from log_analyzer.models import LogEntry, TimelineStats


class TimelineAnalyzer:
    """时间分布分析器"""

    def __init__(self) -> None:
        self.hour_counter: Counter[int] = Counter()
        self.day_counter: Counter[str] = Counter()
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def analyze_entry(self, entry: LogEntry) -> None:
        """分析单条日志"""
        ts = entry.timestamp

        # 按小时统计
        self.hour_counter[ts.hour] += 1

        # 按日期统计
        day_str = ts.strftime("%Y-%m-%d")
        self.day_counter[day_str] += 1

        # 更新时间范围
        if self.start_time is None or ts < self.start_time:
            self.start_time = ts
        if self.end_time is None or ts > self.end_time:
            self.end_time = ts

    def analyze_entries(self, entries: list[LogEntry]) -> None:
        """分析多条日志"""
        for entry in entries:
            self.analyze_entry(entry)

    def get_stats(self) -> TimelineStats:
        """获取统计结果"""
        peak_hour = 0
        peak_count = 0

        if self.hour_counter:
            peak_hour, peak_count = self.hour_counter.most_common(1)[0]

        return TimelineStats(
            by_hour=dict(self.hour_counter),
            by_day=dict(self.day_counter),
            peak_hour=peak_hour,
            peak_count=peak_count,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def reset(self) -> None:
        """重置分析器"""
        self.hour_counter.clear()
        self.day_counter.clear()
        self.start_time = None
        self.end_time = None

    def get_hour_chart(self, width: int = 40) -> str:
        """
        生成小时分布的文本柱状图

        Args:
            width: 图表宽度

        Returns:
            文本图表
        """
        if not self.hour_counter:
            return "No data"

        max_count = max(self.hour_counter.values())
        lines = []

        for hour in range(24):
            count = self.hour_counter.get(hour, 0)
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{hour:02d}:00 | {bar} {count}")

        return "\n".join(lines)

    def get_day_chart(self, width: int = 40) -> str:
        """
        生成日分布的文本柱状图

        Args:
            width: 图表宽度

        Returns:
            文本图表
        """
        if not self.day_counter:
            return "No data"

        max_count = max(self.day_counter.values())
        lines = []

        for day in sorted(self.day_counter.keys()):
            count = self.day_counter[day]
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{day} | {bar} {count}")

        return "\n".join(lines)

