"""数学工具测试

演示基本测试、参数化、异常测试。
"""

import pytest
from testing_lab.math_utils import (
    add,
    subtract,
    multiply,
    divide,
    factorial,
    fibonacci,
    is_prime,
    calculate_average,
    calculate_discount,
)


# ============================================================
# 基本测试
# ============================================================

class TestBasicOperations:
    """基本运算测试"""

    def test_add(self):
        assert add(1, 2) == 3
        assert add(-1, 1) == 0
        assert add(0, 0) == 0

    def test_subtract(self):
        assert subtract(5, 3) == 2
        assert subtract(3, 5) == -2

    def test_multiply(self):
        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0


# ============================================================
# 参数化测试
# ============================================================

class TestParametrized:
    """参数化测试示例"""

    @pytest.mark.parametrize("a,b,expected", [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
        (-5, -3, -8),
    ])
    def test_add_parametrized(self, a, b, expected):
        assert add(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (10, 2, 5.0),
        (9, 3, 3.0),
        (1, 4, 0.25),
        (-10, 2, -5.0),
    ])
    def test_divide_parametrized(self, a, b, expected):
        assert divide(a, b) == expected


# ============================================================
# 异常测试
# ============================================================

class TestExceptions:
    """异常测试示例"""

    def test_divide_by_zero(self):
        with pytest.raises(ValueError):
            divide(1, 0)

    def test_divide_by_zero_message(self):
        with pytest.raises(ValueError, match="除数不能为零"):
            divide(1, 0)

    def test_factorial_negative(self):
        with pytest.raises(ValueError, match="必须是非负整数"):
            factorial(-1)

    def test_empty_list_average(self):
        with pytest.raises(ValueError, match="列表不能为空"):
            calculate_average([])


# ============================================================
# 斐波那契数列测试
# ============================================================

class TestFibonacci:
    """斐波那契数列测试"""

    @pytest.mark.parametrize("n,expected", [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (10, 55),
        (20, 6765),
    ])
    def test_fibonacci(self, n, expected):
        assert fibonacci(n) == expected

    def test_fibonacci_negative(self):
        with pytest.raises(ValueError):
            fibonacci(-1)


# ============================================================
# 质数测试
# ============================================================

class TestIsPrime:
    """质数判断测试"""

    @pytest.mark.parametrize("n", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    def test_primes(self, n):
        assert is_prime(n) is True

    @pytest.mark.parametrize("n", [0, 1, 4, 6, 8, 9, 10, 12, 15, 100])
    def test_non_primes(self, n):
        assert is_prime(n) is False


# ============================================================
# 折扣计算测试
# ============================================================

class TestDiscount:
    """折扣计算测试"""

    def test_no_discount(self):
        assert calculate_discount(100, 0) == 100

    def test_full_discount(self):
        assert calculate_discount(100, 100) == 0

    def test_half_discount(self):
        assert calculate_discount(100, 50) == 50

    @pytest.mark.parametrize("price,discount,expected", [
        (100, 10, 90),
        (200, 25, 150),
        (50, 20, 40),
    ])
    def test_various_discounts(self, price, discount, expected):
        assert calculate_discount(price, discount) == expected

    def test_negative_price(self):
        with pytest.raises(ValueError, match="价格不能为负"):
            calculate_discount(-100, 10)

    def test_invalid_discount(self):
        with pytest.raises(ValueError, match="折扣必须在 0-100 之间"):
            calculate_discount(100, 150)

        with pytest.raises(ValueError, match="折扣必须在 0-100 之间"):
            calculate_discount(100, -10)


# ============================================================
# 使用 Fixture
# ============================================================

class TestWithFixtures:
    """使用 fixture 的测试"""

    def test_average(self, sample_list):
        """使用 conftest.py 中定义的 sample_list fixture"""
        result = calculate_average(sample_list)
        assert result == 3.0

    def test_average_single(self):
        assert calculate_average([5]) == 5.0

    def test_average_floats(self):
        assert calculate_average([1.5, 2.5, 3.0]) == pytest.approx(2.333, rel=0.01)

