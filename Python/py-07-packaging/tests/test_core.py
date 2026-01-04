"""核心功能测试"""

import pytest
from packaging_lab.core import greet, calculate


class TestGreet:
    """greet 函数测试"""

    def test_default_greeting(self):
        """测试默认问候语"""
        result = greet("World")
        assert result == "Hello, World!"

    def test_custom_greeting(self):
        """测试自定义问候语"""
        result = greet("Python", "Hi")
        assert result == "Hi, Python!"

    def test_empty_name(self):
        """测试空名字"""
        result = greet("")
        assert result == "Hello, !"


class TestCalculate:
    """calculate 函数测试"""

    def test_add(self):
        """测试加法"""
        assert calculate(1, 2) == 3.0
        assert calculate(1, 2, "add") == 3.0

    def test_sub(self):
        """测试减法"""
        assert calculate(10, 3, "sub") == 7.0

    def test_mul(self):
        """测试乘法"""
        assert calculate(3, 4, "mul") == 12.0

    def test_div(self):
        """测试除法"""
        assert calculate(10, 2, "div") == 5.0

    def test_div_by_zero(self):
        """测试除以零"""
        with pytest.raises(ZeroDivisionError):
            calculate(1, 0, "div")

    def test_invalid_operation(self):
        """测试无效运算"""
        with pytest.raises(ValueError, match="不支持的运算"):
            calculate(1, 2, "pow")

