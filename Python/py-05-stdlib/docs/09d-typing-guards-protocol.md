# 类型守卫与 Protocol

> TypeGuard、类型窄化和结构化子类型

## TypeGuard - 自定义类型守卫

`TypeGuard` 用于告诉类型检查器函数可以窄化类型。

```python
from typing import TypeGuard

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    """检查是否为字符串列表"""
    return all(isinstance(x, str) for x in val)

def process(data: list[object]) -> None:
    if is_string_list(data):
        # 类型检查器知道 data 现在是 list[str]
        for s in data:
            print(s.upper())  # OK，s 是 str
    else:
        print("Not a string list")
```

### 与 isinstance 的区别

```python
from typing import TypeGuard

# isinstance 自动窄化
def process1(x: int | str) -> None:
    if isinstance(x, str):
        print(x.upper())  # x: str
    else:
        print(x + 1)      # x: int

# TypeGuard 用于复杂情况
def is_positive_int(x: int) -> TypeGuard[int]:
    """检查是否为正整数"""
    return x > 0

def process2(x: int | None) -> None:
    if x is not None and is_positive_int(x):
        # x 被窄化为 int
        print(f"Positive: {x}")
```

### 实际示例

```python
from typing import TypeGuard, Any

# 验证字典结构
def is_user_dict(data: dict[str, Any]) -> TypeGuard[dict[str, str | int]]:
    """检查是否为有效的用户字典"""
    return (
        isinstance(data.get("name"), str) and
        isinstance(data.get("age"), int)
    )

def process_user(data: dict[str, Any]) -> None:
    if is_user_dict(data):
        # data 被窄化
        name: str = data["name"]
        age: int = data["age"]

# 检查可选值
def is_not_none(val: T | None) -> TypeGuard[T]:
    return val is not None

from typing import TypeVar
T = TypeVar("T")

def process_optional(value: str | None) -> str:
    if is_not_none(value):
        return value.upper()  # value: str
    return "default"
```

---

## 类型窄化技巧

### assert isinstance

```python
def process(x: int | str | list[int]) -> None:
    # isinstance 窄化
    if isinstance(x, str):
        print(x.upper())
    elif isinstance(x, list):
        print(sum(x))
    else:
        print(x + 1)

    # assert 也能窄化
    assert isinstance(x, str)
    print(x.upper())  # x: str
```

### 条件表达式

```python
def get_length(x: str | None) -> int:
    # 条件检查窄化
    if x is None:
        return 0
    return len(x)  # x: str

    # 或使用提前返回
    if x is None:
        return 0
    # 此后 x: str
    return len(x)
```

### 类型别名窄化

```python
from typing import Union

Number = Union[int, float]

def process_number(x: Number) -> None:
    if isinstance(x, int):
        # x: int
        print(f"Integer: {x}")
    else:
        # x: float (因为 Number 只有 int 和 float)
        print(f"Float: {x:.2f}")
```

---

## Protocol - 结构化子类型

`Protocol` 定义结构化（鸭子类型）接口，不需要显式继承。

### 基础用法

```python
from typing import Protocol

class Drawable(Protocol):
    """可绘制协议"""
    def draw(self) -> None: ...

# 实现协议（无需继承）
class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

# 两者都满足 Drawable 协议
render(Circle())  # OK
render(Square())  # OK
```

### 带属性的协议

```python
from typing import Protocol

class Named(Protocol):
    """具有 name 属性的协议"""
    name: str

class User:
    def __init__(self, name: str):
        self.name = name

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

def greet(obj: Named) -> None:
    print(f"Hello, {obj.name}")

greet(User("Alice"))      # OK
greet(Product("Book", 10))  # OK
```

### 带方法签名的协议

```python
from typing import Protocol

class Comparable(Protocol):
    """可比较协议"""
    def __lt__(self, other: "Comparable") -> bool: ...
    def __eq__(self, other: object) -> bool: ...

def find_min(items: list[Comparable]) -> Comparable:
    return min(items)

# int 和 str 都实现了 __lt__ 和 __eq__
find_min([3, 1, 2])        # OK
find_min(["c", "a", "b"])  # OK
```

---

## @runtime_checkable

使 Protocol 支持运行时 isinstance 检查。

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Circle")

class NotDrawable:
    pass

# 运行时检查
circle = Circle()
not_drawable = NotDrawable()

print(isinstance(circle, Drawable))       # True
print(isinstance(not_drawable, Drawable))  # False

# 条件处理
def try_draw(obj: object) -> None:
    if isinstance(obj, Drawable):
        obj.draw()  # 类型窄化为 Drawable
    else:
        print("Cannot draw")
```

### 注意事项

```python
@runtime_checkable
class HasName(Protocol):
    name: str

# runtime_checkable 只检查方法是否存在
# 不检查属性类型
class Fake:
    name: int = 123

print(isinstance(Fake(), HasName))  # True！（只检查 name 存在）
```

---

## Protocol vs ABC

| 特性 | Protocol | ABC |
|------|----------|-----|
| 继承要求 | ❌ 不需要 | ✅ 必须继承 |
| 类型检查 | 结构化 | 名义化 |
| 运行时检查 | @runtime_checkable | isinstance |
| 实现方式 | 鸭子类型 | 显式声明 |

```python
from abc import ABC, abstractmethod
from typing import Protocol

# ABC 方式：必须继承
class DrawableABC(ABC):
    @abstractmethod
    def draw(self) -> None: ...

class CircleABC(DrawableABC):  # 必须继承
    def draw(self) -> None:
        print("Circle")

# Protocol 方式：不需要继承
class DrawableProtocol(Protocol):
    def draw(self) -> None: ...

class CircleProtocol:  # 不需要继承
    def draw(self) -> None:
        print("Circle")

def render_abc(shape: DrawableABC) -> None:
    shape.draw()

def render_protocol(shape: DrawableProtocol) -> None:
    shape.draw()

# ABC 需要继承关系
render_abc(CircleABC())      # OK
# render_abc(CircleProtocol())  # 错误！

# Protocol 只看结构
render_protocol(CircleProtocol())  # OK
render_protocol(CircleABC())       # OK（也满足结构）
```

### 何时用哪个？

| 场景 | 推荐 |
|------|------|
| 第三方库类型 | Protocol（无法修改） |
| 内部接口 | 都可以 |
| 需要 isinstance | ABC 或 @runtime_checkable |
| 简单鸭子类型 | Protocol |
| 强制实现约束 | ABC |

---

## 复杂协议示例

### 迭代器协议

```python
from typing import Protocol, TypeVar, Iterator

T = TypeVar("T", covariant=True)

class Iterable(Protocol[T]):
    def __iter__(self) -> Iterator[T]: ...

def first(items: Iterable[T]) -> T | None:
    for item in items:
        return item
    return None

# 列表、元组、集合都满足
first([1, 2, 3])      # int | None
first(("a", "b"))     # str | None
first({1, 2, 3})      # int | None
```

### 上下文管理器协议

```python
from typing import Protocol, TypeVar

T = TypeVar("T", covariant=True)

class ContextManager(Protocol[T]):
    def __enter__(self) -> T: ...
    def __exit__(self, *args) -> None: ...

def use_resource(cm: ContextManager[str]) -> str:
    with cm as value:
        return value
```

### 可调用协议

```python
from typing import Protocol

class Handler(Protocol):
    def __call__(self, event: str, data: dict) -> bool: ...

def process_event(handler: Handler, event: str, data: dict) -> None:
    if handler(event, data):
        print("Event handled")

# 函数满足协议
def my_handler(event: str, data: dict) -> bool:
    print(f"Handling {event}")
    return True

# 类实例也可以
class MyHandler:
    def __call__(self, event: str, data: dict) -> bool:
        return True

process_event(my_handler, "click", {})
process_event(MyHandler(), "click", {})
```

---

## JS/TS 对照

| Python | TypeScript | 说明 |
|--------|------------|------|
| `TypeGuard[T]` | `x is T` | 类型守卫 |
| `Protocol` | `interface` | 结构化类型 |
| `@runtime_checkable` | 无（TS 无运行时类型） | 运行时检查 |
| ABC | `abstract class` | 抽象类 |

```typescript
// TypeScript - 类型守卫
function isString(x: unknown): x is string {
    return typeof x === "string";
}

// TypeScript - 接口（默认结构化）
interface Drawable {
    draw(): void;
}

class Circle {
    draw() { console.log("Circle"); }
}

// Circle 自动满足 Drawable，无需 implements
```

```python
# Python - TypeGuard
from typing import TypeGuard

def is_string(x: object) -> TypeGuard[str]:
    return isinstance(x, str)

# Python - Protocol
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Circle")
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| TypeGuard 返回 False 不窄化 | 只有 True 才窄化 | 在 if 分支使用 |
| Protocol 默认不支持 isinstance | 没有 @runtime_checkable | 添加装饰器 |
| runtime_checkable 只检查方法存在 | 不检查签名和属性类型 | 了解限制 |
| 忘记协议方法的 ... | 需要方法体 | 使用 `def method(self) -> T: ...` |

