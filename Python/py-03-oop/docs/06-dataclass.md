# 06. dataclass

## 🎯 本节目标

- 掌握 @dataclass 装饰器
- 理解 field() 配置
- 创建不可变数据类

---

## 📝 基础用法

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# 自动生成 __init__, __repr__, __eq__
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)

print(p1)        # Point(x=1.0, y=2.0)
print(p1 == p2)  # True
```

### 等价的普通类

```python
# 不用 dataclass 需要手写这些
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
```

---

## ⚙️ dataclass 参数

```python
from dataclasses import dataclass

@dataclass(
    init=True,       # 生成 __init__
    repr=True,       # 生成 __repr__
    eq=True,         # 生成 __eq__
    order=False,     # 生成比较方法（<, <=, >, >=）
    frozen=False,    # 不可变
    slots=False,     # 使用 __slots__（Python 3.10+）
)
class Config:
    name: str
    value: int
```

### order=True

```python
@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
print(v1 < v2)  # True
print(sorted([v2, v1]))  # [Version(1, 2, 3), Version(1, 3, 0)]
```

### frozen=True

```python
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(1.0, 2.0)
# p.x = 3.0  # ❌ FrozenInstanceError
print(hash(p))  # 可哈希（可作为 dict 键）
```

---

## 🔧 field() 配置

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name: str
    age: int = 0                              # 默认值
    tags: List[str] = field(default_factory=list)  # 可变默认值
    _id: int = field(default=0, repr=False)   # 不在 repr 中显示
    created: float = field(default_factory=lambda: __import__('time').time())
```

### field() 参数

| 参数 | 说明 |
|------|------|
| `default` | 默认值 |
| `default_factory` | 默认值工厂（可变类型必须用） |
| `repr` | 是否包含在 repr 中 |
| `compare` | 是否包含在比较中 |
| `hash` | 是否包含在哈希中 |
| `init` | 是否包含在 __init__ 中 |

### ⚠️ 可变默认值陷阱

```python
# ❌ 错误：可变默认值
@dataclass
class BadClass:
    items: list = []  # 会报错！

# ✅ 正确：使用 default_factory
@dataclass
class GoodClass:
    items: list = field(default_factory=list)
```

---

## 🔄 post_init

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)

    def __post_init__(self):
        """在 __init__ 之后自动调用"""
        self.area = self.width * self.height

rect = Rectangle(3, 4)
print(rect.area)  # 12.0
```

---

## 🏷️ 继承

```python
from dataclasses import dataclass

@dataclass
class Animal:
    name: str
    age: int

@dataclass
class Dog(Animal):
    breed: str

    def bark(self):
        return f"{self.name} says Woof!"

dog = Dog("Buddy", 3, "Golden Retriever")
print(dog)  # Dog(name='Buddy', age=3, breed='Golden Retriever')
```

---

## 🆚 对比其他方案

### vs namedtuple

```python
from collections import namedtuple
from dataclasses import dataclass

# namedtuple
Point = namedtuple('Point', ['x', 'y'])

# dataclass
@dataclass
class Point:
    x: float
    y: float
```

| 特性 | dataclass | namedtuple |
|------|-----------|------------|
| 可变性 | 默认可变 | 不可变 |
| 方法定义 | 支持 | 支持 |
| 继承 | 方便 | 麻烦 |
| 默认值 | 方便 | 需要技巧 |
| 内存 | 较大 | 较小 |

### vs TypedDict

```python
from typing import TypedDict

class PersonDict(TypedDict):
    name: str
    age: int

# TypedDict 用于字典类型标注
# dataclass 用于创建数据类
```

### vs pydantic

```python
# pydantic 提供数据验证
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

    # 自动验证类型！
u = User(name="Alice", age="25")  # age 自动转为 int
```

---

## 🛠️ 实用技巧

### 转换为字典

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 25)
print(asdict(p))   # {'name': 'Alice', 'age': 25}
print(astuple(p))  # ('Alice', 25)
```

### 复制并修改

```python
from dataclasses import dataclass, replace

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1.0, 2.0)
p2 = replace(p1, x=3.0)
print(p2)  # Point(x=3.0, y=2.0)
```

### 与 JSON 集成

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class User:
    name: str
    age: int

user = User("Alice", 25)
json_str = json.dumps(asdict(user))
print(json_str)  # {"name": "Alice", "age": 25}
```

---

## ✅ 本节要点

1. `@dataclass` 自动生成 `__init__`, `__repr__`, `__eq__`
2. `frozen=True` 创建不可变类
3. `order=True` 支持排序
4. 可变默认值用 `field(default_factory=...)`
5. `__post_init__` 在初始化后执行
6. `asdict()` 转为字典

