#!/usr/bin/env python3
"""
01_class_basics.py - 类基础演示
"""

# =============================================================================
# 1. 类定义
# =============================================================================
print("=== 类定义 ===")


class Person:
    """人类"""

    # 类属性
    species = "Human"

    def __init__(self, name: str, age: int):
        """构造器"""
        self.name = name  # 实例属性
        self.age = age

    def greet(self) -> str:
        """实例方法"""
        return f"Hello, I'm {self.name}"

    @classmethod
    def create_anonymous(cls) -> "Person":
        """类方法：工厂方法"""
        return cls("Anonymous", 0)

    @staticmethod
    def is_adult(age: int) -> bool:
        """静态方法：工具函数"""
        return age >= 18


# 创建实例
alice = Person("Alice", 25)
print(f"名字: {alice.name}")
print(f"问候: {alice.greet()}")
print(f"物种: {Person.species}")
print(f"是否成年: {Person.is_adult(alice.age)}")

# 使用类方法创建
anon = Person.create_anonymous()
print(f"匿名用户: {anon.name}")

# =============================================================================
# 2. 类属性 vs 实例属性
# =============================================================================
print("\n=== 类属性 vs 实例属性 ===")


class Counter:
    count = 0  # 类属性

    def __init__(self, name: str):
        self.name = name  # 实例属性
        Counter.count += 1


c1 = Counter("A")
c2 = Counter("B")
print(f"创建了 {Counter.count} 个实例")
print(f"c1.name = {c1.name}")

# =============================================================================
# 3. 三种方法类型
# =============================================================================
print("\n=== 三种方法类型 ===")


class Calculator:
    def __init__(self, value: float = 0):
        self.value = value

    def add(self, x: float) -> "Calculator":
        """实例方法：操作实例"""
        self.value += x
        return self

    @classmethod
    def from_string(cls, s: str) -> "Calculator":
        """类方法：工厂方法"""
        return cls(float(s))

    @staticmethod
    def is_number(s: str) -> bool:
        """静态方法：工具函数"""
        try:
            float(s)
            return True
        except ValueError:
            return False


calc = Calculator.from_string("10")
calc.add(5).add(3)
print(f"计算结果: {calc.value}")
print(f"'abc' 是数字? {Calculator.is_number('abc')}")
print(f"'123' 是数字? {Calculator.is_number('123')}")

# =============================================================================
# 4. 访问控制
# =============================================================================
print("\n=== 访问控制 ===")


class BankAccount:
    def __init__(self, balance: float):
        self.balance = balance  # 公开
        self._internal = "internal"  # 约定私有
        self.__secret = "secret"  # 名称改写

    def get_secret(self) -> str:
        return self.__secret


account = BankAccount(100)
print(f"公开: {account.balance}")
print(f"约定私有: {account._internal}")
print(f"通过方法访问: {account.get_secret()}")
print(f"名称改写后: {account._BankAccount__secret}")

# =============================================================================
# 5. 链式调用
# =============================================================================
print("\n=== 链式调用 ===")


class StringBuilder:
    def __init__(self):
        self._parts = []

    def append(self, s: str) -> "StringBuilder":
        self._parts.append(s)
        return self

    def build(self) -> str:
        return "".join(self._parts)


result = StringBuilder().append("Hello").append(" ").append("World").build()
print(f"构建结果: {result}")

print("\n=== 运行完成 ===")

