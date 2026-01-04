#!/usr/bin/env python3
"""
03_magic_methods.py - 魔法方法演示
"""

# =============================================================================
# 1. 字符串表示
# =============================================================================
print("=== 字符串表示 ===")


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


p = Point(1, 2)
print(f"str: {str(p)}")
print(f"repr: {repr(p)}")
print(f"print: {p}")

# =============================================================================
# 2. 运算符重载
# =============================================================================
print("\n=== 运算符重载 ===")


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        return self * scalar

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y)

    def __abs__(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 2 = {v1 * 2}")
print(f"3 * v1 = {3 * v1}")
print(f"-v1 = {-v1}")
print(f"|v1| = {abs(v1)}")

# =============================================================================
# 3. 比较方法
# =============================================================================
print("\n=== 比较方法 ===")

from functools import total_ordering


@total_ordering
class Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __repr__(self) -> str:
        return f"Version({self.major}.{self.minor}.{self.patch})"

    def __eq__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )


v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
v3 = Version(1, 2, 3)
print(f"v1 == v3: {v1 == v3}")
print(f"v1 < v2: {v1 < v2}")
print(f"v2 >= v1: {v2 >= v1}")

# =============================================================================
# 4. 容器协议
# =============================================================================
print("\n=== 容器协议 ===")


class MyList:
    def __init__(self, items=None):
        self._items = list(items) if items else []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]

    def __setitem__(self, index: int, value):
        self._items[index] = value

    def __contains__(self, item) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)


lst = MyList([1, 2, 3, 4, 5])
print(f"长度: {len(lst)}")
print(f"第一个: {lst[0]}")
print(f"3 在列表中? {3 in lst}")
print(f"遍历: {[x for x in lst]}")

# =============================================================================
# 5. 可调用对象
# =============================================================================
print("\n=== 可调用对象 ===")


class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, x: float) -> float:
        return x * self.factor


double = Multiplier(2)
triple = Multiplier(3)
print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")
print(f"callable(double)? {callable(double)}")

# =============================================================================
# 6. 上下文管理器
# =============================================================================
print("\n=== 上下文管理器 ===")

import time


class Timer:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f} 秒")
        return False


with Timer("Sleep 测试"):
    time.sleep(0.1)

print("\n=== 运行完成 ===")

