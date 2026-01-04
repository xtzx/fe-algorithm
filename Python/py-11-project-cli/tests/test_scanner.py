"""Scanner 测试"""

import pytest
from pathlib import Path

from code_counter.scanner import Scanner


class TestScanner:
    """Scanner 测试类"""

    def test_scan_empty_directory(self, tmp_path: Path):
        """测试扫描空目录"""
        scanner = Scanner(tmp_path)
        files = scanner.scan()
        assert files == []

    def test_scan_with_python_files(self, sample_project: Path):
        """测试扫描包含 Python 文件的目录"""
        scanner = Scanner(sample_project)
        files = scanner.scan()

        # 应该找到 Python 和 JS 文件
        extensions = {f.suffix for f in files}
        assert ".py" in extensions
        assert ".js" in extensions

    def test_exclude_patterns(self, sample_project: Path):
        """测试排除模式"""
        scanner = Scanner(sample_project, exclude_patterns=["*.js"])
        files = scanner.scan()

        # 不应该有 JS 文件
        extensions = {f.suffix for f in files}
        assert ".js" not in extensions

    def test_exclude_pycache(self, sample_project: Path):
        """测试默认排除 __pycache__"""
        scanner = Scanner(sample_project)
        files = scanner.scan()

        # 不应该有 __pycache__ 目录中的文件
        for f in files:
            assert "__pycache__" not in str(f)

    def test_directory_not_found(self, tmp_path: Path):
        """测试目录不存在"""
        scanner = Scanner(tmp_path / "nonexistent")

        with pytest.raises(FileNotFoundError):
            scanner.scan()

    def test_not_a_directory(self, python_file: Path):
        """测试路径不是目录"""
        scanner = Scanner(python_file)

        with pytest.raises(NotADirectoryError):
            scanner.scan()

    def test_gitignore_patterns(self, sample_project: Path):
        """测试 .gitignore 模式"""
        # 创建 .gitignore
        gitignore = sample_project / ".gitignore"
        gitignore.write_text("*.js\n")

        scanner = Scanner(sample_project, use_gitignore=True)
        files = scanner.scan()

        # 不应该有 JS 文件
        extensions = {f.suffix for f in files}
        assert ".js" not in extensions

    def test_disable_gitignore(self, sample_project: Path):
        """测试禁用 .gitignore"""
        # 创建 .gitignore
        gitignore = sample_project / ".gitignore"
        gitignore.write_text("*.js\n")

        scanner = Scanner(sample_project, use_gitignore=False)
        files = scanner.scan()

        # 应该有 JS 文件
        extensions = {f.suffix for f in files}
        assert ".js" in extensions

    def test_get_excluded_patterns(self, sample_project: Path):
        """测试获取排除模式"""
        scanner = Scanner(sample_project, exclude_patterns=["custom"])
        patterns = scanner.get_excluded_patterns()

        assert "custom" in patterns
        assert ".git" in patterns  # 默认排除


class TestScannerEdgeCases:
    """Scanner 边界情况测试"""

    def test_scan_hidden_files(self, tmp_path: Path):
        """测试隐藏文件处理"""
        # 创建隐藏文件
        hidden = tmp_path / ".hidden.py"
        hidden.write_text("# hidden")

        normal = tmp_path / "normal.py"
        normal.write_text("# normal")

        # 默认不包含隐藏
        scanner = Scanner(tmp_path, include_hidden=False)
        files = scanner.scan()
        assert len(files) == 1
        assert files[0].name == "normal.py"

        # 包含隐藏
        scanner = Scanner(tmp_path, include_hidden=True)
        files = scanner.scan()
        assert len(files) == 2

    def test_scan_unknown_extension(self, tmp_path: Path):
        """测试未知扩展名"""
        unknown = tmp_path / "file.unknown"
        unknown.write_text("content")

        scanner = Scanner(tmp_path)
        files = scanner.scan()

        # 未知扩展名不应该被包含
        assert len(files) == 0

