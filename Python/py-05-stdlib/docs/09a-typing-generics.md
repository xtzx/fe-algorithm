# 泛型 Generic

> Python 类型系统的泛型支持

## TypeVar 基础

`TypeVar` 用于定义类型变量，类似 TypeScript 的 `<T>`。

```python
from typing import TypeVar

# 定义类型变量
T = TypeVar("T")

def first(items: list[T]) -> T:
    """返回列表第一个元素，保持类型"""
    return items[0]

# 使用
result = first([1, 2, 3])      # result: int
result = first(["a", "b"])     # result: str
```

### TypeVar 命名约定

```python
T = TypeVar("T")           # 通用类型
K = TypeVar("K")           # Key
V = TypeVar("V")           # Value
T_co = TypeVar("T_co", covariant=True)      # 协变
T_contra = TypeVar("T_contra", contravariant=True)  # 逆变
```

---

## TypeVar 约束

### bound - 上界约束

```python
from typing import TypeVar

class Animal:
    def speak(self) -> str:
        return "..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

# T 必须是 Animal 或其子类
T_Animal = TypeVar("T_Animal", bound=Animal)

def make_speak(animal: T_Animal) -> T_Animal:
    print(animal.speak())
    return animal

# 使用
dog = make_speak(Dog())  # dog: Dog
cat = make_speak(Cat())  # cat: Cat
# make_speak("string")   # 类型错误！str 不是 Animal 子类
```

### constraints - 限定类型

```python
from typing import TypeVar

# T 只能是 int 或 str
T = TypeVar("T", int, str)

def double(x: T) -> T:
    return x + x  # int + int 或 str + str

# 使用
double(10)      # 20
double("hi")    # "hihi"
# double(3.14)  # 类型错误！float 不在约束中
```

### bound vs constraints

| 特性 | `bound=X` | `TypeVar("T", A, B)` |
|------|-----------|---------------------|
| 含义 | T 是 X 的子类型 | T 只能是 A 或 B |
| 子类 | 允许 | 不允许 |
| 数量 | 1 个上界 | 多个选项 |

---

## Generic 类

### 基础用法

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    """泛型容器"""

    def __init__(self, item: T):
        self.item = item

    def get(self) -> T:
        return self.item

    def set(self, item: T) -> None:
        self.item = item

# 使用
int_box = Box(42)           # Box[int]
str_box = Box("hello")      # Box[str]

value = int_box.get()       # value: int
int_box.set(100)            # OK
# int_box.set("string")     # 类型错误！
```

### 多类型参数

```python
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

class Pair(Generic[K, V]):
    """键值对"""

    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value

    def get_key(self) -> K:
        return self.key

    def get_value(self) -> V:
        return self.value

# 使用
pair = Pair("name", 42)     # Pair[str, int]
key = pair.get_key()        # key: str
value = pair.get_value()    # value: int
```

### 继承泛型类

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Container(Generic[T]):
    def __init__(self, item: T):
        self.item = item

# 方式 1：保持泛型
class SpecialContainer(Container[T]):
    def process(self) -> T:
        return self.item

# 方式 2：固定类型
class IntContainer(Container[int]):
    def add(self, value: int) -> int:
        return self.item + value
```

---

## 协变与逆变

### 协变 (Covariant)

如果 `Dog` 是 `Animal` 的子类，那么 `Box[Dog]` 是 `Box[Animal]` 的子类。

```python
from typing import TypeVar, Generic

T_co = TypeVar("T_co", covariant=True)

class ReadOnlyBox(Generic[T_co]):
    """只读容器，协变"""

    def __init__(self, item: T_co):
        self._item = item

    def get(self) -> T_co:
        return self._item

    # 注意：协变类型不能出现在参数位置
    # def set(self, item: T_co): ...  # 错误！

def process_animal_box(box: ReadOnlyBox[Animal]) -> None:
    print(box.get().speak())

# Dog 是 Animal 子类，所以 ReadOnlyBox[Dog] 可以传给 ReadOnlyBox[Animal]
dog_box = ReadOnlyBox(Dog())
process_animal_box(dog_box)  # OK！协变允许这样做
```

### 逆变 (Contravariant)

如果 `Dog` 是 `Animal` 的子类，那么 `Handler[Animal]` 是 `Handler[Dog]` 的子类。

```python
from typing import TypeVar, Generic

T_contra = TypeVar("T_contra", contravariant=True)

class Handler(Generic[T_contra]):
    """处理器，逆变"""

    def handle(self, item: T_contra) -> None:
        print(f"Handling {item}")

    # 注意：逆变类型不能出现在返回值位置
    # def get(self) -> T_contra: ...  # 错误！

def use_dog_handler(handler: Handler[Dog]) -> None:
    handler.handle(Dog())

# Animal 是 Dog 的父类，所以 Handler[Animal] 可以传给 Handler[Dog]
animal_handler = Handler[Animal]()
use_dog_handler(animal_handler)  # OK！逆变允许这样做
```

### 协变/逆变总结

| 类型 | 关系 | 适用场景 | 限制 |
|------|------|---------|------|
| 协变 | 子类 → 子类 | 只读/生产者 | 不能作参数 |
| 逆变 | 父类 → 子类 | 只写/消费者 | 不能作返回值 |
| 不变 | 必须相同 | 可读可写 | 无 |

---

## 实用示例

### 栈实现

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

# 使用
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
value = stack.pop()  # value: int
```

### 可选值容器

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Maybe(Generic[T]):
    """可选值容器（类似 Rust 的 Option）"""

    def __init__(self, value: T | None = None):
        self._value = value

    @classmethod
    def some(cls, value: T) -> "Maybe[T]":
        return cls(value)

    @classmethod
    def none(cls) -> "Maybe[T]":
        return cls()

    def is_some(self) -> bool:
        return self._value is not None

    def is_none(self) -> bool:
        return self._value is None

    def unwrap(self) -> T:
        if self._value is None:
            raise ValueError("Called unwrap on None")
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value if self._value is not None else default

    def map(self, fn: "Callable[[T], U]") -> "Maybe[U]":
        if self._value is None:
            return Maybe.none()
        return Maybe.some(fn(self._value))

# 使用
from typing import Callable, TypeVar
U = TypeVar("U")

result = Maybe.some(42)
doubled = result.map(lambda x: x * 2)  # Maybe[int]
value = doubled.unwrap_or(0)  # 84
```

---

## JS/TS 对照

| Python | TypeScript | 说明 |
|--------|------------|------|
| `TypeVar("T")` | `<T>` | 类型变量 |
| `TypeVar("T", bound=X)` | `<T extends X>` | 上界约束 |
| `TypeVar("T", A, B)` | `<T extends A \| B>` | 联合约束 |
| `Generic[T]` | `class X<T>` | 泛型类 |
| `covariant=True` | `out T` (概念) | 协变 |
| `contravariant=True` | `in T` (概念) | 逆变 |

```typescript
// TypeScript
function first<T>(items: T[]): T {
    return items[0];
}

class Box<T> {
    constructor(private item: T) {}
    get(): T { return this.item; }
}
```

```python
# Python
from typing import TypeVar, Generic

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

class Box(Generic[T]):
    def __init__(self, item: T):
        self.item = item
    def get(self) -> T:
        return self.item
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| TypeVar 名不匹配 | `TypeVar("X")` 但变量名是 `T` | 名称要一致 |
| 忘记继承 Generic | 类中使用 TypeVar 但没继承 | 继承 `Generic[T]` |
| 协变用于可写 | 协变类型作为参数 | 使用不变或逆变 |
| 运行时获取泛型参数 | 泛型是编译时概念 | 使用 `__orig_class__` |

