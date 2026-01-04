"""报告器测试"""

import pytest
import json

from data_lab.reporters import generate_report, basic_stats, field_stats


class TestGenerateReport:
    """generate_report 测试"""

    def test_generate_report(self, sample_users):
        """测试生成报告"""
        report = generate_report(sample_users)

        assert report.total_records == 3
        assert report.total_fields == 3
        assert len(report.fields) == 3

    def test_report_to_dict(self, sample_users):
        """测试报告转字典"""
        report = generate_report(sample_users)
        data = report.to_dict()

        assert "summary" in data
        assert "fields" in data
        assert data["summary"]["total_records"] == 3

    def test_report_to_json(self, sample_users):
        """测试报告转 JSON"""
        report = generate_report(sample_users)
        json_str = report.to_json()

        data = json.loads(json_str)
        assert data["summary"]["total_records"] == 3

    def test_report_to_markdown(self, sample_users):
        """测试报告转 Markdown"""
        report = generate_report(sample_users)
        md = report.to_markdown()

        assert "# Data Quality Report" in md
        assert "| Field |" in md

    def test_empty_data(self):
        """测试空数据"""
        report = generate_report([])
        assert report.total_records == 0


class TestBasicStats:
    """basic_stats 测试"""

    def test_basic_stats(self):
        """测试基础统计"""
        values = [1, 2, 3, 4, 5]
        stats = basic_stats(values)

        assert stats["count"] == 5
        assert stats["sum"] == 15
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["mean"] == 3

    def test_empty_values(self):
        """测试空值"""
        stats = basic_stats([])
        assert stats["count"] == 0


class TestFieldStats:
    """field_stats 测试"""

    def test_field_stats(self, sample_users):
        """测试字段统计"""
        stats = field_stats(sample_users, "age")

        assert stats["total"] == 3
        assert stats["non_null"] == 3
        assert stats["null_count"] == 0

    def test_numeric_stats(self, sample_users):
        """测试数值统计"""
        stats = field_stats(sample_users, "age")

        assert "numeric_stats" in stats
        assert stats["numeric_stats"]["min"] == 25
        assert stats["numeric_stats"]["max"] == 35

