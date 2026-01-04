"""良好代码示例

这个模块展示了符合代码质量标准的 Python 代码。
包括：正确的类型注解、良好的格式化、合理的命名等。
"""

from dataclasses import dataclass
from typing import Protocol


# ============================================================
# 类型注解示例
# ============================================================


def add_numbers(a: int, b: int) -> int:
    """两数相加

    Args:
        a: 第一个数字
        b: 第二个数字

    Returns:
        两数之和
    """
    return a + b


def greet(name: str, greeting: str = "Hello") -> str:
    """生成问候语

    Args:
        name: 要问候的名字
        greeting: 问候语前缀

    Returns:
        完整的问候语
    """
    return f"{greeting}, {name}!"


def find_max(numbers: list[int]) -> int | None:
    """找出列表中的最大值

    Args:
        numbers: 整数列表

    Returns:
        最大值，如果列表为空则返回 None
    """
    if not numbers:
        return None
    return max(numbers)


# ============================================================
# 数据类示例
# ============================================================


@dataclass
class User:
    """用户数据类"""

    id: int
    name: str
    email: str
    is_active: bool = True

    def display_name(self) -> str:
        """获取显示名称"""
        return f"{self.name} <{self.email}>"


@dataclass(frozen=True)
class Point:
    """不可变的点"""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """计算到另一个点的距离"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


# ============================================================
# Protocol 示例
# ============================================================


class Drawable(Protocol):
    """可绘制对象的协议"""

    def draw(self) -> str:
        """绘制对象"""
        ...


class Circle:
    """圆形"""

    def __init__(self, radius: float) -> None:
        self.radius = radius

    def draw(self) -> str:
        return f"Circle(radius={self.radius})"


class Rectangle:
    """矩形"""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def draw(self) -> str:
        return f"Rectangle({self.width}x{self.height})"


def render(shape: Drawable) -> None:
    """渲染可绘制对象"""
    print(shape.draw())


# ============================================================
# 列表推导式和生成器
# ============================================================


def get_even_numbers(n: int) -> list[int]:
    """获取前 n 个偶数"""
    return [x for x in range(n) if x % 2 == 0]


def fibonacci(limit: int):
    """斐波那契生成器"""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


# ============================================================
# 上下文管理器
# ============================================================


class Timer:
    """计时器上下文管理器"""

    def __init__(self, name: str = "Timer") -> None:
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "Timer":
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        import time

        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """获取经过的时间"""
        return self.end_time - self.start_time


# ============================================================
# 主函数
# ============================================================


def main() -> None:
    """主函数"""
    # 基本函数
    print(add_numbers(1, 2))
    print(greet("World"))
    print(find_max([1, 5, 3, 9, 2]))

    # 数据类
    user = User(id=1, name="Alice", email="alice@example.com")
    print(user.display_name())

    point1 = Point(0, 0)
    point2 = Point(3, 4)
    print(f"Distance: {point1.distance_to(point2)}")

    # Protocol
    render(Circle(5.0))
    render(Rectangle(10, 20))

    # 生成器
    print(list(fibonacci(100)))

    # 上下文管理器
    with Timer("test") as timer:
        _ = sum(range(1000000))
    print(f"Elapsed: {timer.elapsed:.4f}s")


if __name__ == "__main__":
    main()

