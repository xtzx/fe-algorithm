#!/usr/bin/env python3
"""
05_dataclass.py - dataclass 演示
"""

from dataclasses import dataclass, field, asdict, astuple, replace

# =============================================================================
# 1. 基本 dataclass
# =============================================================================
print("=== 基本 dataclass ===")


@dataclass
class Point:
    x: float
    y: float


p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)
p3 = Point(3.0, 4.0)

print(f"点: {p1}")
print(f"p1 == p2: {p1 == p2}")
print(f"p1 == p3: {p1 == p3}")

# =============================================================================
# 2. 默认值和 field
# =============================================================================
print("\n=== 默认值和 field ===")


@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tags: list[str] = field(default_factory=list)
    _internal: str = field(default="", repr=False)

    @property
    def total_value(self) -> float:
        return self.price * self.quantity


product = Product("Widget", 9.99, 10, ["sale", "new"])
print(f"产品: {product}")
print(f"总价值: {product.total_value}")

# =============================================================================
# 3. 不可变 dataclass
# =============================================================================
print("\n=== 不可变 dataclass ===")


@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float


ip = ImmutablePoint(1.0, 2.0)
print(f"不可变点: {ip}")

try:
    ip.x = 3.0
except Exception as e:
    print(f"修改失败: {type(e).__name__}")

# 可以作为字典键
points = {ip: "origin"}
print(f"作为字典键: {points[ImmutablePoint(1.0, 2.0)]}")

# =============================================================================
# 4. 排序支持
# =============================================================================
print("\n=== 排序支持 ===")


@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int


versions = [Version(1, 2, 3), Version(2, 0, 0), Version(1, 3, 0)]
print(f"排序前: {versions}")
print(f"排序后: {sorted(versions)}")
print(f"v1 < v2: {Version(1, 0, 0) < Version(2, 0, 0)}")

# =============================================================================
# 5. post_init
# =============================================================================
print("\n=== post_init ===")


@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)


rect = Rectangle(3, 4)
print(f"矩形: {rect}")

# =============================================================================
# 6. 转换和复制
# =============================================================================
print("\n=== 转换和复制 ===")


@dataclass
class User:
    name: str
    age: int


user = User("Alice", 25)

# 转换为字典
user_dict = asdict(user)
print(f"字典: {user_dict}")

# 转换为元组
user_tuple = astuple(user)
print(f"元组: {user_tuple}")

# 复制并修改
user2 = replace(user, age=26)
print(f"原用户: {user}")
print(f"新用户: {user2}")

# =============================================================================
# 7. 继承
# =============================================================================
print("\n=== 继承 ===")


@dataclass
class Person:
    name: str
    age: int


@dataclass
class Employee(Person):
    employee_id: str
    department: str


emp = Employee("Bob", 30, "E001", "Engineering")
print(f"员工: {emp}")

print("\n=== 运行完成 ===")

