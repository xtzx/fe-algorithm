"""Counter 测试"""

import pytest
from pathlib import Path

from code_counter.counter import Counter, count_directory
from code_counter.models import FileStats


class TestCounter:
    """Counter 测试类"""

    def test_count_python_file(self, python_file: Path):
        """测试统计 Python 文件"""
        counter = Counter()
        stats = counter.count_file(python_file)

        assert stats.language == "Python"
        assert stats.total_lines > 0
        assert stats.code_lines > 0
        assert stats.comment_lines >= 0
        assert stats.blank_lines >= 0

    def test_count_empty_file(self, tmp_path: Path):
        """测试空文件"""
        empty = tmp_path / "empty.py"
        empty.write_text("")

        counter = Counter()
        stats = counter.count_file(empty)

        assert stats.total_lines == 0
        assert stats.code_lines == 0
        assert stats.comment_lines == 0
        assert stats.blank_lines == 0

    def test_count_only_comments(self, tmp_path: Path):
        """测试只有注释的文件"""
        comments_only = tmp_path / "comments.py"
        comments_only.write_text("""# Comment 1
# Comment 2
# Comment 3
""")

        counter = Counter()
        stats = counter.count_file(comments_only)

        assert stats.comment_lines == 3
        assert stats.code_lines == 0

    def test_count_only_blanks(self, tmp_path: Path):
        """测试只有空行的文件"""
        blanks_only = tmp_path / "blanks.py"
        blanks_only.write_text("\n\n\n\n")

        counter = Counter()
        stats = counter.count_file(blanks_only)

        assert stats.blank_lines == 4
        assert stats.code_lines == 0
        assert stats.comment_lines == 0

    def test_count_javascript(self, tmp_path: Path):
        """测试 JavaScript 文件"""
        js_file = tmp_path / "test.js"
        js_file.write_text("""// Single line comment
function foo() {
    /* Multi-line
       comment */
    return 1;
}
""")

        counter = Counter()
        stats = counter.count_file(js_file)

        assert stats.language == "JavaScript"
        assert stats.comment_lines >= 2

    def test_count_multiline_docstring(self, tmp_path: Path):
        """测试多行文档字符串"""
        py_file = tmp_path / "docstring.py"
        py_file.write_text('''"""
This is a
multi-line
docstring.
"""

def foo():
    pass
''')

        counter = Counter()
        stats = counter.count_file(py_file)

        # 文档字符串应该被计为注释
        assert stats.comment_lines >= 4


class TestCountFiles:
    """count_files 测试"""

    def test_count_multiple_files(self, sample_project: Path):
        """测试统计多个文件"""
        counter = Counter()

        from code_counter.scanner import Scanner
        scanner = Scanner(sample_project)
        files = scanner.scan()

        result = counter.count_files(files, sample_project)

        assert result.total_files > 0
        assert len(result.by_language) > 0

    def test_count_directory_convenience(self, sample_project: Path):
        """测试 count_directory 便捷函数"""
        result = count_directory(sample_project)

        assert result.total_files > 0
        assert "Python" in result.by_language


class TestFileStats:
    """FileStats 测试"""

    def test_file_stats_validation(self, tmp_path: Path):
        """测试 FileStats 验证"""
        with pytest.raises(ValueError):
            FileStats(
                path=tmp_path / "test.py",
                language="Python",
                total_lines=-1,
            )

    def test_file_stats_properties(self, python_file: Path):
        """测试 FileStats 属性"""
        stats = FileStats(
            path=python_file,
            language="Python",
            total_lines=10,
            code_lines=5,
            comment_lines=3,
            blank_lines=2,
        )

        assert stats.total_lines == 10
        assert stats.code_lines == 5


class TestScanResult:
    """ScanResult 测试"""

    def test_scan_result_properties(self, sample_project: Path):
        """测试 ScanResult 属性"""
        result = count_directory(sample_project)

        assert result.total_files >= 0
        assert result.total_lines >= 0
        assert result.total_code_lines >= 0
        assert result.total_comment_lines >= 0
        assert result.total_blank_lines >= 0

    def test_scan_result_add_error(self, sample_project: Path):
        """测试添加错误"""
        result = count_directory(sample_project)
        result.add_error("Test error")

        assert result.error_count == 1
        assert "Test error" in result.errors

