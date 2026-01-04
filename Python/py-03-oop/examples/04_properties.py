#!/usr/bin/env python3
"""
04_properties.py - 属性演示
"""

# =============================================================================
# 1. 基本 property
# =============================================================================
print("=== 基本 property ===")


class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self) -> float:
        """getter"""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """setter"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self) -> float:
        """只读计算属性"""
        import math

        return math.pi * self._radius**2


circle = Circle(5)
print(f"半径: {circle.radius}")
print(f"面积: {circle.area:.2f}")

circle.radius = 10
print(f"新半径: {circle.radius}")
print(f"新面积: {circle.area:.2f}")

try:
    circle.radius = -1
except ValueError as e:
    print(f"错误: {e}")

# =============================================================================
# 2. 属性验证
# =============================================================================
print("\n=== 属性验证 ===")


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("Name must be non-empty string")
        self._name = value.strip()

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Age must be non-negative integer")
        self._age = value


person = Person("Alice", 25)
print(f"姓名: {person.name}, 年龄: {person.age}")

try:
    person.age = -1
except ValueError as e:
    print(f"验证错误: {e}")

# =============================================================================
# 3. 温度转换
# =============================================================================
print("\n=== 温度转换 ===")


class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self._celsius = (value - 32) * 5 / 9

    @property
    def kelvin(self) -> float:
        return self._celsius + 273.15


temp = Temperature(100)
print(f"摄氏: {temp.celsius}°C")
print(f"华氏: {temp.fahrenheit}°F")
print(f"开尔文: {temp.kelvin}K")

temp.fahrenheit = 32
print(f"\n设置华氏 32°F 后:")
print(f"摄氏: {temp.celsius}°C")

# =============================================================================
# 4. 缓存属性
# =============================================================================
print("\n=== 缓存属性 ===")

from functools import cached_property


class DataProcessor:
    def __init__(self, data: list):
        self.data = data

    @cached_property
    def expensive_result(self) -> float:
        """只计算一次"""
        print("正在计算...")
        import time

        time.sleep(0.1)  # 模拟耗时操作
        return sum(self.data) / len(self.data)


processor = DataProcessor([1, 2, 3, 4, 5])
print(f"第一次访问: {processor.expensive_result}")
print(f"第二次访问: {processor.expensive_result}")  # 不会重新计算

# =============================================================================
# 5. 只读属性
# =============================================================================
print("\n=== 只读属性 ===")


class Configuration:
    def __init__(self, settings: dict):
        self._settings = settings.copy()

    @property
    def settings(self) -> dict:
        """返回副本，防止修改"""
        return self._settings.copy()


config = Configuration({"debug": True, "port": 8080})
settings = config.settings
settings["debug"] = False
print(f"原配置: {config.settings}")  # 不受影响

print("\n=== 运行完成 ===")

