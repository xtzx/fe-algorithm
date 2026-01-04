# 05. 抽象类与协议

## 🎯 本节目标

- 掌握抽象基类 ABC
- 理解 Protocol（结构化子类型）
- 对比鸭子类型与显式接口

---

## 📝 抽象基类 ABC

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """抽象类：不能直接实例化"""

    @abstractmethod
    def area(self) -> float:
        """抽象方法：子类必须实现"""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self) -> str:
        """普通方法：子类可以继承"""
        return f"Shape with area {self.area():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# Shape()  # ❌ TypeError: Can't instantiate abstract class
rect = Rectangle(3, 4)
print(rect.area())       # 12
print(rect.describe())   # Shape with area 12.00
```

### 抽象属性

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @property
    @abstractmethod
    def species(self) -> str:
        pass

    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    @property
    def species(self) -> str:
        return "Canis familiaris"

    def speak(self) -> str:
        return "Woof!"
```

---

## 🦆 鸭子类型

Python 传统风格：如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子。

```python
# 不需要继承，只要有相同方法就行
class Duck:
    def walk(self):
        return "Duck walking"

    def quack(self):
        return "Quack!"

class Robot:
    def walk(self):
        return "Robot walking"

    def quack(self):
        return "Beep boop quack!"

def make_it_quack(thing):
    """接受任何有 quack 方法的对象"""
    return thing.quack()

print(make_it_quack(Duck()))   # Quack!
print(make_it_quack(Robot()))  # Beep boop quack!
```

---

## 📋 typing.Protocol

Python 3.8+ 引入的结构化子类型，结合了鸭子类型的灵活性和类型检查。

```python
from typing import Protocol

class Drawable(Protocol):
    """协议：定义接口"""
    def draw(self) -> str:
        ...

class Circle:
    """没有显式继承 Drawable"""
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    """类型检查器认可任何实现 draw 的类"""
    print(shape.draw())

render(Circle())  # ✅ 类型检查通过
render(Square())  # ✅ 类型检查通过
```

### Protocol vs ABC

| 特性 | ABC | Protocol |
|------|-----|----------|
| 检查时机 | 运行时 | 静态类型检查 |
| 需要继承 | 是 | 否 |
| 方法验证 | 实例化时 | 类型检查时 |
| 适用场景 | 强制接口 | 结构化类型 |

---

## 🔄 runtime_checkable

让 Protocol 支持 `isinstance` 检查：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None:
        ...

class File:
    def close(self) -> None:
        print("File closed")

class Connection:
    def close(self) -> None:
        print("Connection closed")

# 运行时检查
print(isinstance(File(), Closeable))  # True
print(isinstance("string", Closeable))  # False
```

---

## 📊 标准库中的协议

```python
from typing import Iterable, Iterator, Callable, Sized

# Iterable：可迭代（有 __iter__）
def process(items: Iterable[int]) -> int:
    return sum(items)

# Sized：有长度（有 __len__）
def show_length(obj: Sized) -> int:
    return len(obj)

# Callable：可调用
def apply(func: Callable[[int], int], value: int) -> int:
    return func(value)
```

---

## 🎯 选择指南

### 何时用 ABC

1. 需要强制子类实现某些方法
2. 需要运行时检查
3. 有共享的实现代码

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def save(self, entity): pass

    @abstractmethod
    def find(self, id): pass

    def find_or_create(self, id, factory):
        """共享实现"""
        entity = self.find(id)
        if entity is None:
            entity = factory()
            self.save(entity)
        return entity
```

### 何时用 Protocol

1. 定义接口但不强制继承
2. 与现有代码兼容
3. 静态类型检查足够

```python
from typing import Protocol

class Logger(Protocol):
    def log(self, message: str) -> None: ...

# 任何有 log 方法的类都可以
def use_logger(logger: Logger):
    logger.log("Hello")
```

### 何时用鸭子类型

1. 简单场景
2. 不需要类型检查
3. 灵活性优先

```python
def stringify(obj):
    """任何有 __str__ 的对象"""
    return str(obj)
```

---

## ✅ 本节要点

1. `ABC` + `@abstractmethod` 定义抽象类
2. 抽象类不能实例化
3. `Protocol` 提供结构化子类型（不需要继承）
4. `@runtime_checkable` 让 Protocol 支持 isinstance
5. 鸭子类型最灵活，ABC 最严格

