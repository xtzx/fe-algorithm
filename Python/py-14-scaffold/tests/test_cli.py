"""
测试 CLI 模块
"""

import pytest
from unittest.mock import patch

from scaffold.cli import main, create_parser


class TestCreateParser:
    """测试命令行解析器"""

    def test_parser_creation(self):
        """测试解析器创建"""
        parser = create_parser()

        assert parser.prog == "scaffold"

    def test_verbose_flag(self):
        """测试 verbose 参数"""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "version"])

        assert args.verbose is True

    def test_quiet_flag(self):
        """测试 quiet 参数"""
        parser = create_parser()
        args = parser.parse_args(["--quiet", "version"])

        assert args.quiet is True

    def test_json_log_flag(self):
        """测试 json-log 参数"""
        parser = create_parser()
        args = parser.parse_args(["--json-log", "version"])

        assert args.json_log is True


class TestVersionCommand:
    """测试 version 命令"""

    def test_version(self, capsys):
        """测试显示版本"""
        result = main(["version"])

        assert result == 0

        captured = capsys.readouterr()
        assert "scaffold version" in captured.out


class TestConfigCommand:
    """测试 config 命令"""

    def test_config_text(self, capsys):
        """测试文本格式配置输出"""
        result = main(["config"])

        assert result == 0

        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "app_name" in captured.out

    def test_config_json(self, capsys):
        """测试 JSON 格式配置输出"""
        result = main(["config", "--json"])

        assert result == 0

        captured = capsys.readouterr()
        assert "{" in captured.out
        assert "app_name" in captured.out


class TestRunCommand:
    """测试 run 命令"""

    def test_run_basic(self):
        """测试基本运行"""
        result = main(["run"])

        assert result == 0

    def test_run_with_config(self, tmp_path):
        """测试带配置文件运行"""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")

        result = main(["run", "--config", str(config_file)])

        assert result == 0


class TestInitCommand:
    """测试 init 命令"""

    def test_init_new_project(self, tmp_path, monkeypatch):
        """测试初始化新项目"""
        monkeypatch.chdir(tmp_path)

        result = main(["init", "my-project"])

        # 目前 init 命令只是一个占位符
        assert result == 0

    def test_init_existing_directory(self, tmp_path, monkeypatch):
        """测试初始化已存在的目录"""
        monkeypatch.chdir(tmp_path)

        # 创建已存在的目录
        (tmp_path / "existing-project").mkdir()

        result = main(["init", "existing-project"])

        assert result == 1  # 应该失败


class TestMainHelp:
    """测试帮助输出"""

    def test_no_command(self, capsys):
        """测试无命令时显示帮助"""
        result = main([])

        assert result == 0

        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "commands" in captured.out.lower()

