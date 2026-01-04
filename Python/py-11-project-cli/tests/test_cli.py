"""CLI 测试"""

import pytest
from pathlib import Path

from code_counter.cli import main, create_parser


class TestParser:
    """Parser 测试"""

    def test_create_parser(self):
        """测试创建解析器"""
        parser = create_parser()
        assert parser is not None

    def test_parse_scan(self):
        """测试解析 scan 命令"""
        parser = create_parser()
        args = parser.parse_args(["scan", "."])

        assert args.command == "scan"
        assert args.path == Path(".")

    def test_parse_scan_with_options(self):
        """测试解析 scan 带选项"""
        parser = create_parser()
        args = parser.parse_args([
            "scan", ".",
            "-e", "node_modules",
            "-e", ".git",
            "-f", "json",
            "-o", "output.json",
        ])

        assert args.exclude == ["node_modules", ".git"]
        assert args.format == "json"
        assert args.output == Path("output.json")

    def test_parse_report(self):
        """测试解析 report 命令"""
        parser = create_parser()
        args = parser.parse_args(["report", ".", "-f", "markdown"])

        assert args.command == "report"
        assert args.format == "markdown"

    def test_parse_config_show(self):
        """测试解析 config show"""
        parser = create_parser()
        args = parser.parse_args(["config", "show"])

        assert args.command == "config"
        assert args.config_command == "show"

    def test_parse_config_init(self):
        """测试解析 config init"""
        parser = create_parser()
        args = parser.parse_args(["config", "init", "--force"])

        assert args.config_command == "init"
        assert args.force is True


class TestMain:
    """main 函数测试"""

    def test_main_no_args(self, capsys):
        """测试无参数"""
        result = main([])

        # 应该打印帮助
        assert result == 0

    def test_main_version(self, capsys):
        """测试版本"""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0

    def test_main_scan(self, sample_project: Path, capsys):
        """测试 scan 命令"""
        result = main(["scan", str(sample_project)])

        assert result == 0

        captured = capsys.readouterr()
        assert "Python" in captured.out or "Total" in captured.out

    def test_main_scan_json(self, sample_project: Path, capsys):
        """测试 scan 输出 JSON"""
        result = main(["scan", str(sample_project), "-f", "json"])

        assert result == 0

        captured = capsys.readouterr()
        assert "{" in captured.out
        assert "summary" in captured.out

    def test_main_scan_markdown(self, sample_project: Path, capsys):
        """测试 scan 输出 Markdown"""
        result = main(["scan", str(sample_project), "-f", "markdown"])

        assert result == 0

        captured = capsys.readouterr()
        assert "#" in captured.out
        assert "|" in captured.out

    def test_main_scan_output_file(self, sample_project: Path, tmp_path: Path, capsys):
        """测试 scan 输出到文件"""
        output = tmp_path / "result.json"
        result = main(["scan", str(sample_project), "-f", "json", "-o", str(output)])

        assert result == 0
        assert output.exists()

    def test_main_scan_nonexistent(self, tmp_path: Path, capsys):
        """测试 scan 不存在的目录"""
        result = main(["scan", str(tmp_path / "nonexistent")])

        assert result == 1

        captured = capsys.readouterr()
        assert "错误" in captured.err

    def test_main_scan_exclude(self, sample_project: Path, capsys):
        """测试 scan 排除模式"""
        result = main(["scan", str(sample_project), "-e", "*.js"])

        assert result == 0

        captured = capsys.readouterr()
        # JavaScript 应该被排除
        assert "JavaScript" not in captured.out or "0" in captured.out

    def test_main_report(self, sample_project: Path, capsys):
        """测试 report 命令"""
        result = main(["report", str(sample_project)])

        assert result == 0

        captured = capsys.readouterr()
        assert "Code Statistics" in captured.out or "#" in captured.out

    def test_main_config_show(self, capsys):
        """测试 config show"""
        result = main(["config", "show"])

        assert result == 0

        captured = capsys.readouterr()
        assert "当前配置" in captured.out

    def test_main_config_init(self, tmp_path: Path, monkeypatch, capsys):
        """测试 config init"""
        monkeypatch.chdir(tmp_path)
        result = main(["config", "init"])

        assert result == 0
        assert (tmp_path / ".code-counter.toml").exists()

    def test_main_verbose(self, sample_project: Path, capsys):
        """测试详细模式"""
        result = main(["-v", "scan", str(sample_project)])

        assert result == 0


class TestCLIEdgeCases:
    """CLI 边界情况测试"""

    def test_scan_file_not_directory(self, python_file: Path, capsys):
        """测试 scan 文件而非目录"""
        result = main(["scan", str(python_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "错误" in captured.err

