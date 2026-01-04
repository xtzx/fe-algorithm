"""报告器测试"""

from datetime import datetime
from pathlib import Path

import pytest

from log_analyzer.models import (
    AnalysisReport,
    ErrorStats,
    RequestStats,
    TimelineStats,
)
from log_analyzer.reporters import JsonReporter, MarkdownReporter, TerminalReporter


@pytest.fixture
def sample_report() -> AnalysisReport:
    """示例分析报告"""
    return AnalysisReport(
        generated_at=datetime(2024, 1, 1, 12, 0, 0),
        files_analyzed=5,
        total_entries=1000,
        valid_entries=980,
        invalid_entries=20,
        error_stats=ErrorStats(
            total_errors=50,
            total_warnings=30,
            total_critical=10,
            by_level={"ERROR": 50, "WARNING": 30, "CRITICAL": 10},
            by_hour={12: 40, 13: 30, 14: 20},
            top_messages=[
                ("Connection timeout", 25),
                ("Database error", 15),
                ("Auth failed", 10),
            ],
        ),
        request_stats=RequestStats(
            total_requests=500,
            by_status_code={200: 400, 404: 50, 500: 50},
            by_method={"GET": 350, "POST": 150},
            top_urls=[
                ("/api/users", 200),
                ("/api/products", 150),
                ("/api/orders", 100),
            ],
            avg_response_time=0.15,
            max_response_time=2.5,
            error_rate=10.0,
        ),
        timeline_stats=TimelineStats(
            by_hour={12: 300, 13: 400, 14: 300},
            by_day={"2024-01-01": 500, "2024-01-02": 500},
            peak_hour=13,
            peak_count=400,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 2, 14, 0, 0),
        ),
    )


class TestTerminalReporter:
    """终端报告器测试"""

    def test_generate_with_colors(self, sample_report: AnalysisReport) -> None:
        reporter = TerminalReporter(use_colors=True)
        output = reporter.generate(sample_report)

        assert "日志分析报告" in output
        assert "错误统计" in output
        assert "请求统计" in output
        assert "时间分布" in output

    def test_generate_without_colors(self, sample_report: AnalysisReport) -> None:
        reporter = TerminalReporter(use_colors=False)
        output = reporter.generate(sample_report)

        # 不应该包含 ANSI 转义码
        assert "\033[" not in output

    def test_error_section(self, sample_report: AnalysisReport) -> None:
        reporter = TerminalReporter(use_colors=False)
        output = reporter.generate(sample_report)

        assert "ERROR" in output
        assert "WARNING" in output
        assert "CRITICAL" in output

    def test_request_section(self, sample_report: AnalysisReport) -> None:
        reporter = TerminalReporter(use_colors=False)
        output = reporter.generate(sample_report)

        assert "500" in output
        assert "/api/users" in output


class TestJsonReporter:
    """JSON 报告器测试"""

    def test_generate(self, sample_report: AnalysisReport) -> None:
        reporter = JsonReporter()
        output = reporter.generate(sample_report)

        import json
        data = json.loads(output)

        assert data["files_analyzed"] == 5
        assert data["total_entries"] == 1000
        assert "error_stats" in data

    def test_save(self, sample_report: AnalysisReport, tmp_path: Path) -> None:
        reporter = JsonReporter()
        output_path = tmp_path / "report.json"

        result_path = reporter.save(sample_report, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "files_analyzed" in content


class TestMarkdownReporter:
    """Markdown 报告器测试"""

    def test_generate(self, sample_report: AnalysisReport) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(sample_report)

        assert "# 日志分析报告" in output
        assert "## 📋 概览" in output
        assert "## ⚠️ 错误统计" in output
        assert "## 📊 请求统计" in output
        assert "## 📈 时间分布" in output

    def test_table_format(self, sample_report: AnalysisReport) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(sample_report)

        # 应该包含 Markdown 表格
        assert "|---" in output
        assert "| 指标 | 数值 |" in output

    def test_save(self, sample_report: AnalysisReport, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        output_path = tmp_path / "report.md"

        result_path = reporter.save(sample_report, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "# 日志分析报告" in content

