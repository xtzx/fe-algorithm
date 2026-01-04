#!/usr/bin/env python3
"""typing 模块演示"""

from typing import (
    Optional, Union, Any, Callable,
    TypeVar, Generic, Literal, Final,
    TypedDict, Protocol
)
from dataclasses import dataclass


def demo_basic_types():
    """基础类型"""
    print("=" * 50)
    print("1. 基础类型")
    print("=" * 50)

    # 基本类型注解
    name: str = "Alice"
    age: int = 25
    height: float = 1.75
    is_active: bool = True
    nothing: None = None

    print(f"name: {name} ({type(name).__name__})")
    print(f"age: {age} ({type(age).__name__})")
    print(f"height: {height} ({type(height).__name__})")
    print(f"is_active: {is_active} ({type(is_active).__name__})")
    print(f"nothing: {nothing}")

    # 函数类型注解
    def greet(name: str) -> str:
        return f"Hello, {name}"

    print(f"greet('World'): {greet('World')}")


def demo_container_types():
    """容器类型"""
    print("\n" + "=" * 50)
    print("2. 容器类型 (Python 3.9+)")
    print("=" * 50)

    # 直接使用内置类型
    names: list[str] = ["Alice", "Bob"]
    scores: dict[str, int] = {"Alice": 90, "Bob": 85}
    unique: set[str] = {"a", "b", "c"}
    point: tuple[int, int] = (10, 20)
    values: tuple[int, ...] = (1, 2, 3, 4, 5)

    print(f"names: {names}")
    print(f"scores: {scores}")
    print(f"unique: {unique}")
    print(f"point: {point}")
    print(f"values: {values}")


def demo_optional_union():
    """Optional 和 Union"""
    print("\n" + "=" * 50)
    print("3. Optional 和 Union")
    print("=" * 50)

    # Optional - 可能为 None
    def find_user(user_id: int) -> Optional[str]:
        users = {1: "Alice", 2: "Bob"}
        return users.get(user_id)

    print(f"find_user(1): {find_user(1)}")
    print(f"find_user(99): {find_user(99)}")

    # Union - 多种类型之一
    def process_value(value: Union[int, str]) -> str:
        return str(value)

    print(f"process_value(42): {process_value(42)}")
    print(f"process_value('hello'): {process_value('hello')}")

    # Python 3.10+ 语法
    def new_find(user_id: int) -> str | None:
        return find_user(user_id)

    print(f"new_find(2): {new_find(2)}")


def demo_callable():
    """Callable 类型"""
    print("\n" + "=" * 50)
    print("4. Callable 类型")
    print("=" * 50)

    # 函数类型
    def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
        return func(a, b)

    def add(x: int, y: int) -> int:
        return x + y

    def multiply(x: int, y: int) -> int:
        return x * y

    print(f"apply(add, 3, 4): {apply(add, 3, 4)}")
    print(f"apply(multiply, 3, 4): {apply(multiply, 3, 4)}")

    # 无参数函数
    def run_callback(callback: Callable[[], None]) -> None:
        callback()

    run_callback(lambda: print("Callback executed!"))


def demo_typevar():
    """TypeVar 泛型"""
    print("\n" + "=" * 50)
    print("5. TypeVar 泛型")
    print("=" * 50)

    T = TypeVar("T")

    def first(items: list[T]) -> T:
        return items[0]

    # 使用
    int_first = first([1, 2, 3])
    str_first = first(["a", "b", "c"])

    print(f"first([1, 2, 3]): {int_first}")
    print(f"first(['a', 'b', 'c']): {str_first}")

    # 约束类型
    Number = TypeVar("Number", int, float)

    def add_numbers(a: Number, b: Number) -> Number:
        return a + b

    print(f"add_numbers(1, 2): {add_numbers(1, 2)}")
    print(f"add_numbers(1.5, 2.5): {add_numbers(1.5, 2.5)}")


def demo_generic():
    """Generic 泛型类"""
    print("\n" + "=" * 50)
    print("6. Generic 泛型类")
    print("=" * 50)

    T = TypeVar("T")

    class Box(Generic[T]):
        def __init__(self, item: T) -> None:
            self.item = item

        def get(self) -> T:
            return self.item

        def __repr__(self) -> str:
            return f"Box({self.item!r})"

    # 使用
    int_box: Box[int] = Box(42)
    str_box: Box[str] = Box("hello")

    print(f"int_box: {int_box}, get(): {int_box.get()}")
    print(f"str_box: {str_box}, get(): {str_box.get()}")


def demo_literal():
    """Literal 字面量类型"""
    print("\n" + "=" * 50)
    print("7. Literal 字面量类型")
    print("=" * 50)

    def set_log_level(level: Literal["debug", "info", "warning", "error"]) -> None:
        print(f"设置日志级别: {level}")

    set_log_level("info")
    set_log_level("error")
    # set_log_level("unknown")  # 类型检查器会报错


def demo_final():
    """Final 常量"""
    print("\n" + "=" * 50)
    print("8. Final 常量")
    print("=" * 50)

    MAX_SIZE: Final = 100
    API_URL: Final[str] = "https://api.example.com"

    print(f"MAX_SIZE: {MAX_SIZE}")
    print(f"API_URL: {API_URL}")

    # MAX_SIZE = 200  # 类型检查器会报错


def demo_typeddict():
    """TypedDict 结构化字典"""
    print("\n" + "=" * 50)
    print("9. TypedDict 结构化字典")
    print("=" * 50)

    class User(TypedDict):
        name: str
        age: int
        email: str

    user: User = {
        "name": "Alice",
        "age": 25,
        "email": "alice@example.com"
    }

    print(f"User: {user}")
    print(f"name: {user['name']}")

    # 可选字段
    class UserOptional(TypedDict, total=False):
        name: str
        age: int
        nickname: str  # 可选

    user2: UserOptional = {"name": "Bob", "age": 30}
    print(f"UserOptional: {user2}")


def demo_protocol():
    """Protocol 结构化子类型"""
    print("\n" + "=" * 50)
    print("10. Protocol 结构化子类型")
    print("=" * 50)

    class Readable(Protocol):
        def read(self) -> str: ...

    class Writable(Protocol):
        def write(self, data: str) -> None: ...

    # 任何有 read 方法的类都符合 Readable
    class MyFile:
        def __init__(self, content: str):
            self._content = content

        def read(self) -> str:
            return self._content

    def process_readable(reader: Readable) -> str:
        return reader.read()

    # 不需要显式继承 Readable
    my_file = MyFile("Hello from MyFile!")
    result = process_readable(my_file)
    print(f"process_readable result: {result}")


def demo_practical():
    """实际应用"""
    print("\n" + "=" * 50)
    print("11. 实际应用")
    print("=" * 50)

    @dataclass
    class Config:
        debug: bool = False
        log_level: Literal["debug", "info", "warning", "error"] = "info"
        max_retries: int = 3

    def fetch_data(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        config: Config | None = None
    ) -> dict[str, Any]:
        """模拟获取数据"""
        return {
            "url": url,
            "headers": headers or {},
            "timeout": timeout,
            "config": config
        }

    result = fetch_data(
        "https://api.example.com",
        headers={"Authorization": "Bearer token"},
        config=Config(debug=True)
    )
    print(f"fetch_data result: {result}")


if __name__ == "__main__":
    demo_basic_types()
    demo_container_types()
    demo_optional_union()
    demo_callable()
    demo_typevar()
    demo_generic()
    demo_literal()
    demo_final()
    demo_typeddict()
    demo_protocol()
    demo_practical()

    print("\n✅ typing 演示完成!")


