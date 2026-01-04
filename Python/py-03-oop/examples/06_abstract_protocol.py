#!/usr/bin/env python3
"""
06_abstract_protocol.py - 抽象类与协议演示
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import math

# =============================================================================
# 1. 抽象基类
# =============================================================================
print("=== 抽象基类 ===")


class Shape(ABC):
    """抽象类"""

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self) -> str:
        return f"Shape with area {self.area():.2f}"


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


# 不能实例化抽象类
try:
    s = Shape()
except TypeError as e:
    print(f"抽象类不能实例化: {e}")

# 子类可以实例化
rect = Rectangle(3, 4)
circle = Circle(5)
print(f"矩形面积: {rect.area()}")
print(f"圆形面积: {circle.area():.2f}")
print(f"描述: {rect.describe()}")

# =============================================================================
# 2. 多态
# =============================================================================
print("\n=== 多态 ===")

shapes: list[Shape] = [Rectangle(3, 4), Circle(5), Rectangle(2, 6)]
for shape in shapes:
    print(f"  {shape.__class__.__name__}: 面积={shape.area():.2f}")

# =============================================================================
# 3. Protocol (结构化子类型)
# =============================================================================
print("\n=== Protocol ===")


class Drawable(Protocol):
    """协议：定义接口"""

    def draw(self) -> str: ...


class Square:
    """没有显式继承 Drawable"""

    def __init__(self, size: float):
        self.size = size

    def draw(self) -> str:
        return f"Drawing square with size {self.size}"


class Triangle:
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def draw(self) -> str:
        return f"Drawing triangle with base {self.base}"


def render(shape: Drawable) -> None:
    """接受任何实现 draw 的对象"""
    print(f"  {shape.draw()}")


print("渲染图形:")
render(Square(5))
render(Triangle(3, 4))

# =============================================================================
# 4. runtime_checkable
# =============================================================================
print("\n=== runtime_checkable ===")


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


class Connection:
    def close(self) -> None:
        print("Connection closed")


class FileHandle:
    def close(self) -> None:
        print("File closed")


class NotCloseable:
    pass


print(f"Connection 是 Closeable? {isinstance(Connection(), Closeable)}")
print(f"FileHandle 是 Closeable? {isinstance(FileHandle(), Closeable)}")
print(f"NotCloseable 是 Closeable? {isinstance(NotCloseable(), Closeable)}")

# =============================================================================
# 5. 鸭子类型
# =============================================================================
print("\n=== 鸭子类型 ===")


class Duck:
    def walk(self) -> str:
        return "Duck walking"

    def quack(self) -> str:
        return "Quack!"


class Robot:
    def walk(self) -> str:
        return "Robot walking"

    def quack(self) -> str:
        return "Beep boop quack!"


def make_it_quack(thing) -> None:
    """接受任何有 quack 方法的对象"""
    print(f"  {thing.quack()}")


print("让它们叫:")
make_it_quack(Duck())
make_it_quack(Robot())

print("\n=== 运行完成 ===")

