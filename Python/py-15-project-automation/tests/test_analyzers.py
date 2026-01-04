"""分析器测试"""

from log_analyzer.analyzers import ErrorAnalyzer, RequestAnalyzer, TimelineAnalyzer
from log_analyzer.models import AppLogEntry, NginxLogEntry


class TestErrorAnalyzer:
    """错误分析器测试"""

    def test_analyze_errors(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = ErrorAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        assert stats.total_errors == 1
        assert stats.total_warnings == 1
        assert stats.total_critical == 1

    def test_by_level(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = ErrorAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        assert "ERROR" in stats.by_level
        assert "WARNING" in stats.by_level
        assert "CRITICAL" in stats.by_level

    def test_by_hour(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = ErrorAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        # 所有条目都在 12 点
        assert 12 in stats.by_hour

    def test_reset(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = ErrorAnalyzer()
        analyzer.analyze_entries(sample_app_entries)
        analyzer.reset()

        stats = analyzer.get_stats()
        assert stats.total_errors == 0


class TestRequestAnalyzer:
    """请求分析器测试"""

    def test_analyze_requests(self, sample_nginx_entries: list[NginxLogEntry]) -> None:
        analyzer = RequestAnalyzer()
        analyzer.analyze_entries(sample_nginx_entries)

        stats = analyzer.get_stats()
        assert stats.total_requests == 3

    def test_by_status_code(self, sample_nginx_entries: list[NginxLogEntry]) -> None:
        analyzer = RequestAnalyzer()
        analyzer.analyze_entries(sample_nginx_entries)

        stats = analyzer.get_stats()
        assert 200 in stats.by_status_code
        assert 401 in stats.by_status_code
        assert 500 in stats.by_status_code

    def test_by_method(self, sample_nginx_entries: list[NginxLogEntry]) -> None:
        analyzer = RequestAnalyzer()
        analyzer.analyze_entries(sample_nginx_entries)

        stats = analyzer.get_stats()
        assert stats.by_method["GET"] == 2
        assert stats.by_method["POST"] == 1

    def test_error_rate(self, sample_nginx_entries: list[NginxLogEntry]) -> None:
        analyzer = RequestAnalyzer()
        analyzer.analyze_entries(sample_nginx_entries)

        stats = analyzer.get_stats()
        # 1 out of 3 is 500 error
        assert stats.error_rate == pytest.approx(33.33, rel=0.1)

    def test_response_time(self, sample_nginx_entries: list[NginxLogEntry]) -> None:
        analyzer = RequestAnalyzer()
        analyzer.analyze_entries(sample_nginx_entries)

        stats = analyzer.get_stats()
        assert stats.max_response_time == 1.5


class TestTimelineAnalyzer:
    """时间分布分析器测试"""

    def test_analyze_timeline(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = TimelineAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        assert 12 in stats.by_hour
        assert stats.by_hour[12] == 4

    def test_peak_hour(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = TimelineAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        assert stats.peak_hour == 12
        assert stats.peak_count == 4

    def test_time_range(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = TimelineAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        stats = analyzer.get_stats()
        assert stats.start_time is not None
        assert stats.end_time is not None
        assert stats.start_time <= stats.end_time

    def test_get_hour_chart(self, sample_app_entries: list[AppLogEntry]) -> None:
        analyzer = TimelineAnalyzer()
        analyzer.analyze_entries(sample_app_entries)

        chart = analyzer.get_hour_chart()
        assert "12:00" in chart
        assert "█" in chart


# 需要导入 pytest 以使用 approx
import pytest

