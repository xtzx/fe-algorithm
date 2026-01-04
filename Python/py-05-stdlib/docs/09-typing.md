# 09. typing - 类型提示

## 本节目标

- 掌握基础类型提示
- 学会使用泛型类型
- 了解高级类型

---

## 基础类型

```python
# 基本类型
name: str = "Alice"
age: int = 25
height: float = 1.75
is_active: bool = True
nothing: None = None

# 函数类型提示
def greet(name: str) -> str:
    return f"Hello, {name}"

def process(data: bytes) -> None:
    pass
```

---

## 容器类型

### Python 3.9+（推荐）

```python
# 直接使用内置类型
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 90, "Bob": 85}
unique: set[str] = {"a", "b"}
point: tuple[int, int] = (10, 20)

# 可变长元组
values: tuple[int, ...] = (1, 2, 3, 4)
```

### Python 3.8 及以下

```python
from typing import List, Dict, Set, Tuple

names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 90}
unique: Set[str] = {"a", "b"}
point: Tuple[int, int] = (10, 20)
```

---

## Optional 和 Union

```python
from typing import Optional, Union

# Optional - 可能为 None
def find_user(id: int) -> Optional[str]:
    # 返回 str 或 None
    return None

# 等价于
def find_user(id: int) -> str | None:  # Python 3.10+
    return None

# Union - 多种类型之一
def process(value: Union[int, str]) -> None:
    pass

# Python 3.10+
def process(value: int | str) -> None:
    pass
```

---

## Any

```python
from typing import Any

# 任意类型（尽量避免使用）
def handle(data: Any) -> Any:
    return data
```

---

## Callable

```python
from typing import Callable

# 函数类型
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# 无参数
def run(callback: Callable[[], None]) -> None:
    callback()

# 任意参数
def wrapper(func: Callable[..., str]) -> str:
    return func()
```

---

## TypeVar - 泛型

```python
from typing import TypeVar, Sequence

T = TypeVar("T")

def first(items: Sequence[T]) -> T:
    return items[0]

# 使用
first([1, 2, 3])      # 返回 int
first(["a", "b"])     # 返回 str

# 约束类型
T = TypeVar("T", int, float)

def add(a: T, b: T) -> T:
    return a + b
```

---

## Generic - 泛型类

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

    def get(self) -> T:
        return self.item

# 使用
box_int: Box[int] = Box(42)
box_str: Box[str] = Box("hello")
```

---

## Literal

```python
from typing import Literal

# 只能是特定值
def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass

set_mode("read")   # OK
set_mode("delete") # 类型检查器报错

# 联合字面量
Direction = Literal["north", "south", "east", "west"]
```

---

## Final

```python
from typing import Final

# 常量（不可重新赋值）
MAX_SIZE: Final = 100
API_URL: Final[str] = "https://api.example.com"

MAX_SIZE = 200  # 类型检查器报错
```

---

## TypedDict

```python
from typing import TypedDict

# 定义字典结构
class User(TypedDict):
    name: str
    age: int
    email: str

user: User = {
    "name": "Alice",
    "age": 25,
    "email": "alice@example.com"
}

# 可选字段
class UserOptional(TypedDict, total=False):
    name: str
    age: int
    nickname: str  # 可选
```

---

## Protocol - 结构化子类型

```python
from typing import Protocol

class Readable(Protocol):
    def read(self) -> str:
        ...

class Writeable(Protocol):
    def write(self, data: str) -> None:
        ...

# 任何有 read 方法的对象都可以
def process(reader: Readable) -> str:
    return reader.read()

class MyFile:
    def read(self) -> str:
        return "content"

# 不需要显式继承 Readable
process(MyFile())  # OK
```

---

## 类型别名

```python
from typing import TypeAlias

# 简单别名
UserId = int
UserName = str

# Python 3.10+
Vector: TypeAlias = list[float]

# 复杂类型别名
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
```

---

## 常用模式

### 返回自身类型

```python
from typing import TypeVar, Type

T = TypeVar("T", bound="Animal")

class Animal:
    @classmethod
    def create(cls: Type[T]) -> T:
        return cls()

class Dog(Animal):
    pass

dog: Dog = Dog.create()
```

### 回调和装饰器

```python
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def decorator(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper
```

---

## 运行时 vs 静态检查

```python
from typing import TYPE_CHECKING

# 只在类型检查时导入（避免循环导入）
if TYPE_CHECKING:
    from .module import SomeClass

def func(obj: "SomeClass") -> None:  # 使用字符串避免运行时错误
    pass
```

---

## 类型检查工具

```bash
# mypy
pip install mypy
mypy script.py

# pyright (VS Code Pylance)
pip install pyright
pyright script.py
```

---

## 本节要点

1. 基础类型：`str`, `int`, `float`, `bool`, `None`
2. 容器类型：`list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]`
3. `Optional[T]` = `T | None`
4. `Union[A, B]` = `A | B`
5. `Callable[[Args], Return]` 函数类型
6. `TypeVar` 和 `Generic` 泛型
7. `Literal` 字面量类型
8. `TypedDict` 结构化字典
9. `Protocol` 结构化子类型


