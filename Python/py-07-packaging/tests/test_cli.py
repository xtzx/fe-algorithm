"""CLI 测试"""

from click.testing import CliRunner

from packaging_lab import __version__
from packaging_lab.cli import main


class TestCLI:
    """CLI 命令测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.runner = CliRunner()

    def test_version(self):
        """测试版本显示"""
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_hello_basic(self):
        """测试基本问候"""
        result = self.runner.invoke(main, ["hello", "World"])
        assert result.exit_code == 0
        assert "Hello, World!" in result.output

    def test_hello_custom_greeting(self):
        """测试自定义问候语"""
        result = self.runner.invoke(main, ["hello", "Python", "-g", "Hi"])
        assert result.exit_code == 0
        assert "Hi, Python!" in result.output

    def test_hello_times(self):
        """测试重复次数"""
        result = self.runner.invoke(main, ["hello", "World", "-n", "3"])
        assert result.exit_code == 0
        assert result.output.count("Hello, World!") == 3

    def test_calc_add(self):
        """测试加法计算"""
        result = self.runner.invoke(main, ["calc", "1", "2"])
        assert result.exit_code == 0
        assert "3.0" in result.output

    def test_calc_sub(self):
        """测试减法计算"""
        result = self.runner.invoke(main, ["calc", "10", "3", "-o", "sub"])
        assert result.exit_code == 0
        assert "7.0" in result.output

    def test_calc_div_zero(self):
        """测试除以零"""
        result = self.runner.invoke(main, ["calc", "1", "0", "-o", "div"])
        assert result.exit_code == 1
        assert "除以零" in result.output

    def test_info(self):
        """测试信息显示"""
        result = self.runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "Packaging Lab" in result.output
        assert __version__ in result.output

