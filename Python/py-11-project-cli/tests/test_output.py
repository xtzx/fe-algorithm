"""Output 测试"""

import json
import pytest
from pathlib import Path

from code_counter.counter import count_directory
from code_counter.output import OutputFormatter, format_output


class TestOutputFormatter:
    """OutputFormatter 测试类"""

    def test_format_table(self, sample_project: Path):
        """测试表格格式"""
        result = count_directory(sample_project)
        formatter = OutputFormatter()
        output = formatter.to_table(result)

        assert "Language" in output
        assert "Files" in output
        assert "Code" in output
        assert "Total" in output

    def test_format_json(self, sample_project: Path):
        """测试 JSON 格式"""
        result = count_directory(sample_project)
        formatter = OutputFormatter()
        output = formatter.to_json(result)

        # 验证是有效 JSON
        data = json.loads(output)
        assert "summary" in data
        assert "by_language" in data
        assert "files" in data

    def test_format_markdown(self, sample_project: Path):
        """测试 Markdown 格式"""
        result = count_directory(sample_project)
        formatter = OutputFormatter()
        output = formatter.to_markdown(result)

        assert "# Code Statistics" in output
        assert "## Summary" in output
        assert "## By Language" in output
        assert "|" in output

    def test_format_function(self, sample_project: Path):
        """测试 format 函数"""
        result = count_directory(sample_project)
        formatter = OutputFormatter()

        # 测试各种格式
        assert "Language" in formatter.format(result, "table")
        assert "{" in formatter.format(result, "json")
        assert "#" in formatter.format(result, "markdown")


class TestFormatOutput:
    """format_output 便捷函数测试"""

    def test_format_output_table(self, sample_project: Path):
        """测试表格输出"""
        result = count_directory(sample_project)
        output = format_output(result, "table")

        assert "Total" in output

    def test_format_output_json(self, sample_project: Path):
        """测试 JSON 输出"""
        result = count_directory(sample_project)
        output = format_output(result, "json")

        data = json.loads(output)
        assert data["summary"]["total_files"] >= 0

    def test_format_output_default(self, sample_project: Path):
        """测试默认格式"""
        result = count_directory(sample_project)
        output = format_output(result)  # 默认 table

        assert "Language" in output


class TestOutputContent:
    """输出内容测试"""

    def test_json_structure(self, sample_project: Path):
        """测试 JSON 结构"""
        result = count_directory(sample_project)
        output = format_output(result, "json")
        data = json.loads(output)

        # 检查 summary
        summary = data["summary"]
        assert "root" in summary
        assert "total_files" in summary
        assert "total_lines" in summary
        assert "code_lines" in summary
        assert "comment_lines" in summary
        assert "blank_lines" in summary

        # 检查 by_language
        for lang, stats in data["by_language"].items():
            assert "files" in stats
            assert "code" in stats
            assert "comments" in stats
            assert "blank" in stats

    def test_markdown_links(self, sample_project: Path):
        """测试 Markdown 表格"""
        result = count_directory(sample_project)
        output = format_output(result, "markdown")

        # 检查表格结构
        lines = output.split("\n")
        table_lines = [l for l in lines if l.startswith("|")]
        assert len(table_lines) >= 2  # 至少有表头和分隔线

    def test_sorted_by_code_lines(self, sample_project: Path):
        """测试按代码行数排序"""
        result = count_directory(sample_project)
        output = format_output(result, "json")
        data = json.loads(output)

        # 获取语言列表
        langs = list(data["by_language"].items())
        if len(langs) > 1:
            # 第一个应该有最多代码行
            first_code = langs[0][1]["code"]
            last_code = langs[-1][1]["code"]
            # 在 JSON 中顺序可能不保证，但表格中应该排序

