# 高级类型

> Literal、TypedDict、Final 等高级类型特性

## Literal - 字面量类型

限定变量只能是特定的字面值。

```python
from typing import Literal

# 只能是这几个值
Direction = Literal["north", "south", "east", "west"]

def move(direction: Direction) -> None:
    print(f"Moving {direction}")

move("north")   # OK
move("south")   # OK
# move("up")    # 类型错误！

# 数字字面量
def set_priority(level: Literal[1, 2, 3]) -> None:
    ...

set_priority(1)   # OK
# set_priority(4)  # 类型错误！
```

### 与枚举的对比

```python
from enum import Enum
from typing import Literal

# 枚举方式
class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"

def process_enum(status: Status) -> None: ...
process_enum(Status.PENDING)

# Literal 方式
StatusLit = Literal["pending", "active", "done"]

def process_literal(status: StatusLit) -> None: ...
process_literal("pending")

# 区别：
# - 枚举是运行时对象，Literal 只是类型提示
# - 枚举可以有方法和属性
# - Literal 使用更简洁
```

### 实际应用

```python
from typing import Literal, overload

# HTTP 方法
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

def request(method: HttpMethod, url: str) -> dict:
    ...

# 排序方向
def sort_data(
    data: list[int],
    order: Literal["asc", "desc"] = "asc"
) -> list[int]:
    return sorted(data, reverse=(order == "desc"))
```

---

## TypedDict - 结构化字典

定义字典的结构和键的类型。

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

# 使用
user: User = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}

# 类型检查
user["name"]    # str
user["age"]     # int
# user["phone"]  # 类型错误！键不存在
```

### 可选键

```python
from typing import TypedDict, Required, NotRequired

# Python 3.11+
class Config(TypedDict):
    host: str                    # 必需
    port: int                    # 必需
    debug: NotRequired[bool]     # 可选

# 或使用 total=False
class PartialConfig(TypedDict, total=False):
    host: str     # 可选
    port: int     # 可选
    debug: bool   # 可选

# 混合使用
class MixedConfig(TypedDict, total=False):
    debug: bool   # 可选

class FullConfig(MixedConfig):
    host: str     # 必需
    port: int     # 必需
```

### 继承

```python
from typing import TypedDict

class BaseUser(TypedDict):
    id: int
    name: str

class AdminUser(BaseUser):
    role: str
    permissions: list[str]

admin: AdminUser = {
    "id": 1,
    "name": "Admin",
    "role": "admin",
    "permissions": ["read", "write", "delete"]
}
```

### 与 Pydantic 的区别

```python
from typing import TypedDict
from pydantic import BaseModel

# TypedDict: 只是类型提示，无运行时验证
class UserDict(TypedDict):
    name: str
    age: int

# Pydantic: 有运行时验证
class UserModel(BaseModel):
    name: str
    age: int

# TypedDict
user_dict: UserDict = {"name": "Alice", "age": "30"}  # 运行时不报错！

# Pydantic
user_model = UserModel(name="Alice", age="30")  # 运行时会转换/报错
```

---

## Final - 常量标注

标记变量不应被重新赋值。

```python
from typing import Final

# 常量
MAX_SIZE: Final = 100
API_URL: Final[str] = "https://api.example.com"

# 类型检查器会警告重新赋值
# MAX_SIZE = 200  # 错误！不应修改 Final 变量
```

### 类中使用

```python
from typing import Final

class Config:
    # 类级别常量
    VERSION: Final = "1.0.0"

    def __init__(self):
        # 实例级别常量（只能在 __init__ 中赋值一次）
        self.name: Final = "MyApp"
        # self.name = "Other"  # 错误！

# Config.VERSION = "2.0.0"  # 错误！
```

---

## ClassVar - 类变量

标记变量属于类而非实例。

```python
from typing import ClassVar

class Counter:
    # 类变量
    count: ClassVar[int] = 0

    def __init__(self):
        Counter.count += 1
        # 实例变量
        self.id = Counter.count

c1 = Counter()
c2 = Counter()
print(Counter.count)  # 2
```

### ClassVar vs Final

| 特性 | `ClassVar` | `Final` |
|------|-----------|---------|
| 含义 | 类变量 | 不可变 |
| 可修改 | ✅ 可以 | ❌ 不可以 |
| 实例访问 | 通过类访问 | 直接访问 |

```python
from typing import ClassVar, Final

class Example:
    class_var: ClassVar[int] = 0        # 可修改的类变量
    final_class: Final[str] = "const"   # 不可修改的值

    # 组合使用
    MAX_COUNT: ClassVar[Final[int]] = 100  # 不可修改的类变量
```

---

## Self（Python 3.11+）

返回当前类类型，用于链式调用和工厂方法。

```python
from typing import Self

class Builder:
    def __init__(self):
        self._value = 0

    def add(self, n: int) -> Self:
        self._value += n
        return self

    def multiply(self, n: int) -> Self:
        self._value *= n
        return self

    def build(self) -> int:
        return self._value

# 链式调用
result = Builder().add(5).multiply(2).build()  # 10

# 继承时也能正确工作
class AdvancedBuilder(Builder):
    def subtract(self, n: int) -> Self:
        self._value -= n
        return self

# 返回 AdvancedBuilder 而不是 Builder
adv = AdvancedBuilder().add(10).subtract(3)  # AdvancedBuilder
```

### Python 3.10 及之前的替代方案

```python
from typing import TypeVar

T = TypeVar("T", bound="Builder")

class Builder:
    def add(self, n: int) -> T:  # type: ignore
        ...

    # 或者使用字符串
    def add(self, n: int) -> "Builder":
        ...
```

---

## TypeAlias - 类型别名

显式声明类型别名。

```python
from typing import TypeAlias

# 显式类型别名
UserId: TypeAlias = int
UserName: TypeAlias = str
UserDict: TypeAlias = dict[str, int | str]

# 复杂类型别名
JsonValue: TypeAlias = (
    str | int | float | bool | None |
    list["JsonValue"] | dict[str, "JsonValue"]
)

def parse_json(data: str) -> JsonValue:
    import json
    return json.loads(data)
```

### 与普通赋值的区别

```python
# 普通赋值 - 类型检查器可能不理解这是别名
MyList = list[int]

# 显式别名 - 清楚表明意图
from typing import TypeAlias
MyList: TypeAlias = list[int]
```

---

## NewType - 新类型

创建语义不同但底层类型相同的类型。

```python
from typing import NewType

# 创建新类型
UserId = NewType("UserId", int)
OrderId = NewType("OrderId", int)

def get_user(user_id: UserId) -> dict:
    ...

def get_order(order_id: OrderId) -> dict:
    ...

# 使用
user_id = UserId(123)
order_id = OrderId(456)

get_user(user_id)    # OK
get_order(order_id)  # OK
# get_user(order_id)  # 类型错误！虽然都是 int，但语义不同
# get_user(123)       # 类型错误！需要 UserId 类型
```

### NewType vs TypeAlias

```python
from typing import NewType, TypeAlias

# TypeAlias: 只是别名，完全等价
MyInt: TypeAlias = int
x: MyInt = 42
y: int = x      # OK，完全等价

# NewType: 新类型，不能混用
UserId = NewType("UserId", int)
user_id: UserId = UserId(42)
# z: int = user_id  # 类型错误！（虽然运行时是 int）
```

---

## @overload - 函数重载

定义多个函数签名。

```python
from typing import overload, Literal

@overload
def process(data: str) -> str: ...

@overload
def process(data: int) -> int: ...

@overload
def process(data: list[str]) -> list[str]: ...

def process(data: str | int | list[str]) -> str | int | list[str]:
    """实际实现"""
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return data * 2
    else:
        return [s.upper() for s in data]

# 类型检查器知道：
result1 = process("hello")   # str
result2 = process(42)        # int
result3 = process(["a", "b"])  # list[str]
```

### 根据参数返回不同类型

```python
from typing import overload, Literal

@overload
def fetch(url: str, as_json: Literal[True]) -> dict: ...

@overload
def fetch(url: str, as_json: Literal[False]) -> str: ...

@overload
def fetch(url: str) -> str: ...

def fetch(url: str, as_json: bool = False) -> dict | str:
    response = requests.get(url)
    if as_json:
        return response.json()
    return response.text

# 类型检查器知道：
data = fetch("https://api.example.com", as_json=True)   # dict
text = fetch("https://api.example.com", as_json=False)  # str
text = fetch("https://api.example.com")                 # str
```

---

## JS/TS 对照

| Python | TypeScript | 说明 |
|--------|------------|------|
| `Literal["a", "b"]` | `"a" \| "b"` | 字面量类型 |
| `TypedDict` | `interface` / `type` | 结构化类型 |
| `Final` | `readonly` / `const` | 不可变 |
| `ClassVar` | `static` | 类变量 |
| `Self` | `this` 类型 | 自身类型 |
| `TypeAlias` | `type X = ...` | 类型别名 |
| `NewType` | 品牌类型 (branded) | 新类型 |
| `@overload` | 函数重载 | 多签名 |

```typescript
// TypeScript
type Direction = "north" | "south" | "east" | "west";
interface User { name: string; age: number; }
class Counter { static count: number = 0; }
```

```python
# Python
from typing import Literal, TypedDict, ClassVar

Direction = Literal["north", "south", "east", "west"]
class User(TypedDict): name: str; age: int
class Counter: count: ClassVar[int] = 0
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| TypedDict 无运行时检查 | 只是类型提示 | 需要验证时用 Pydantic |
| NewType 运行时是原类型 | `UserId(1)` 就是 `1` | 只用于类型检查 |
| Final 无运行时强制 | 仍可修改 | 依赖类型检查器 |
| overload 只定义不实现 | 实现函数也要有 | 最后写实现函数 |

