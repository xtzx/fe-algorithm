#!/usr/bin/env python3
"""
文本统计器测试
"""

import unittest
from pathlib import Path
import tempfile
import os

from main import analyze_text, analyze_file


class TestAnalyzeText(unittest.TestCase):
    """测试 analyze_text 函数"""

    def test_empty_text(self):
        """测试空文本"""
        result = analyze_text("")
        self.assertEqual(result["lines"], 0)
        self.assertEqual(result["words"], 0)
        self.assertEqual(result["characters"], 0)

    def test_single_line(self):
        """测试单行文本"""
        result = analyze_text("Hello World")
        self.assertEqual(result["lines"], 1)
        self.assertEqual(result["words"], 2)
        self.assertEqual(result["characters"], 11)

    def test_multiple_lines(self):
        """测试多行文本"""
        text = "Line 1\nLine 2\nLine 3"
        result = analyze_text(text)
        self.assertEqual(result["lines"], 3)
        self.assertEqual(result["words"], 6)

    def test_longest_line(self):
        """测试最长行检测"""
        text = "Short\nThis is a longer line\nMedium line"
        result = analyze_text(text)
        self.assertEqual(result["longest_line"], "This is a longer line")
        self.assertEqual(result["longest_line_length"], 21)

    def test_chinese_text(self):
        """测试中文文本"""
        text = "你好世界\nPython 编程"
        result = analyze_text(text)
        self.assertEqual(result["lines"], 2)
        # 中文分词比较特殊，这里只检查行数

    def test_only_whitespace(self):
        """测试只有空白的行"""
        text = "Hello\n   \nWorld"
        result = analyze_text(text)
        self.assertEqual(result["lines"], 3)
        self.assertEqual(result["words"], 2)  # 空白行不算单词


class TestAnalyzeFile(unittest.TestCase):
    """测试 analyze_file 函数"""

    def setUp(self):
        """创建临时测试文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_file.write_text("Hello World\nPython is great", encoding="utf-8")

    def tearDown(self):
        """清理临时文件"""
        if self.test_file.exists():
            self.test_file.unlink()
        os.rmdir(self.temp_dir)

    def test_analyze_existing_file(self):
        """测试分析存在的文件"""
        result = analyze_file(str(self.test_file))
        self.assertEqual(result["lines"], 2)
        self.assertEqual(result["words"], 5)

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            analyze_file("nonexistent_file.txt")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_real_world_example(self):
        """测试真实场景"""
        text = """The quick brown fox jumps over the lazy dog.
Python is a powerful programming language.
It is widely used in web development, data science, and automation.
This is the last line."""

        result = analyze_text(text)

        self.assertEqual(result["lines"], 4)
        self.assertGreater(result["words"], 0)
        self.assertGreater(result["characters"], 0)
        self.assertGreater(result["longest_line_length"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

