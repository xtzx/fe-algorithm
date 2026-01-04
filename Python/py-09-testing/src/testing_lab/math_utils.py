"""数学工具模块 - 用于演示基本测试"""

from typing import List


def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


def subtract(a: int, b: int) -> int:
    """两数相减"""
    return a - b


def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b


def divide(a: float, b: float) -> float:
    """两数相除

    Raises:
        ValueError: 除数为零时抛出
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


def factorial(n: int) -> int:
    """计算阶乘

    Raises:
        ValueError: n 为负数时抛出
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """计算第 n 个斐波那契数

    Args:
        n: 序号（从 0 开始）
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """判断是否为质数"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def calculate_average(numbers: List[float]) -> float:
    """计算平均值

    Raises:
        ValueError: 列表为空时抛出
    """
    if not numbers:
        raise ValueError("列表不能为空")
    return sum(numbers) / len(numbers)


def calculate_discount(price: float, discount_percent: float) -> float:
    """计算折扣后价格

    Args:
        price: 原价
        discount_percent: 折扣百分比 (0-100)

    Raises:
        ValueError: 价格为负或折扣不在 0-100 之间
    """
    if price < 0:
        raise ValueError("价格不能为负")
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("折扣必须在 0-100 之间")
    return price * (1 - discount_percent / 100)

