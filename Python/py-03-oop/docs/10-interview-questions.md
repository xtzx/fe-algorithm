# 10. 面试高频问题

> 10 个 Python OOP 面试高频问题

---

## 1. Python 中如何实现私有属性？

<details>
<summary>参考答案</summary>

Python 没有真正的私有属性，靠命名约定：

| 命名 | 含义 | 访问 |
|------|------|------|
| `name` | 公开 | 任意访问 |
| `_name` | 约定私有 | 可访问，但不推荐 |
| `__name` | 名称改写 | 变成 `_ClassName__name` |

```python
class Example:
    def __init__(self):
        self.public = "public"
        self._protected = "protected"
        self.__private = "private"

obj = Example()
print(obj.public)      # public
print(obj._protected)  # protected（可访问）
print(obj.__private)   # ❌ AttributeError
print(obj._Example__private)  # private（名称改写后可访问）
```

**单下划线 vs 双下划线**：
- `_name`：约定内部使用，不会被改写
- `__name`：防止子类意外覆盖，会被改写为 `_ClassName__name`

</details>

---

## 2. `__new__` 和 `__init__` 的区别？

<details>
<summary>参考答案</summary>

| 方法 | 作用 | 返回值 | 调用时机 |
|------|------|--------|---------|
| `__new__` | 创建实例 | 必须返回实例 | 在 `__init__` 之前 |
| `__init__` | 初始化实例 | None | 在 `__new__` 之后 |

```python
class Singleton:
    _instance = None

    def __new__(cls):
        print("__new__ called")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        print("__init__ called")

s = Singleton()
# __new__ called
# __init__ called
```

**`__new__` 的用途**：
1. 单例模式
2. 不可变类型（如自定义 str/int）
3. 元类

</details>

---

## 3. 类方法和静态方法的区别？

<details>
<summary>参考答案</summary>

| 类型 | 装饰器 | 第一个参数 | 用途 |
|------|--------|-----------|------|
| 实例方法 | 无 | `self` | 操作实例 |
| 类方法 | `@classmethod` | `cls` | 工厂方法，操作类 |
| 静态方法 | `@staticmethod` | 无 | 工具函数 |

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_str):
        """类方法：工厂方法"""
        parts = date_str.split("-")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    @staticmethod
    def is_valid_date(year, month, day):
        """静态方法：不需要类或实例"""
        return 1 <= month <= 12 and 1 <= day <= 31

date = Date.from_string("2024-01-15")
print(Date.is_valid_date(2024, 1, 15))  # True
```

**何时用类方法**：需要访问类本身（如创建实例）
**何时用静态方法**：与类相关但不需要访问类或实例

</details>

---

## 4. MRO 是什么？如何查看？

<details>
<summary>参考答案</summary>

**MRO（Method Resolution Order）**：方法解析顺序，决定多继承时方法查找顺序。

Python 使用 **C3 线性化算法**。

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

# 查看 MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

print(D.mro())  # 同上
```

**MRO 规则**：
1. 子类优先于父类
2. 父类按声明顺序
3. 保证每个类只出现一次

</details>

---

## 5. `__str__` 和 `__repr__` 的区别？

<details>
<summary>参考答案</summary>

| 方法 | 用途 | 调用时机 | 面向对象 |
|------|------|---------|---------|
| `__str__` | 用户友好 | `str()`, `print()` | 终端用户 |
| `__repr__` | 开发者友好 | `repr()`, 交互式环境 | 开发者 |

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

p = Point(1, 2)
print(str(p))   # (1, 2)
print(repr(p))  # Point(x=1, y=2)
print(p)        # (1, 2)  ← 调用 __str__
```

**最佳实践**：
- `__repr__` 应该返回能重建对象的字符串
- 如果只实现一个，实现 `__repr__`

</details>

---

## 6. 如何让自定义对象可以用 `len()`？

<details>
<summary>参考答案</summary>

实现 `__len__` 方法：

```python
class MyCollection:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

col = MyCollection([1, 2, 3])
print(len(col))  # 3
```

**相关协议**：
- `__len__`：支持 `len()`
- `__bool__`：支持 `bool()`（默认基于 `__len__`）
- `__contains__`：支持 `in` 操作

</details>

---

## 7. 如何让自定义对象可以用 for 遍历？

<details>
<summary>参考答案</summary>

实现 `__iter__` 方法，返回迭代器：

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

for i in Countdown(5):
    print(i)  # 5, 4, 3, 2, 1
```

**或者同时实现 `__iter__` 和 `__next__`**：

```python
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1
```

</details>

---

## 8. dataclass 和普通 class 的区别？

<details>
<summary>参考答案</summary>

**dataclass 自动生成**：
- `__init__`
- `__repr__`
- `__eq__`
- 可选：`__hash__`, 比较方法

```python
# 普通类
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age})"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

# dataclass（等价）
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
```

**dataclass 优势**：
1. 代码更简洁
2. 自动生成常用方法
3. 支持类型注解
4. `frozen=True` 创建不可变类

</details>

---

## 9. Python 的多继承有什么问题？如何解决？

<details>
<summary>参考答案</summary>

**问题：菱形继承（钻石问题）**

```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")
        super().method()

class C(A):
    def method(self):
        print("C")
        super().method()

class D(B, C):
    pass

D().method()  # B, C, A（每个只调用一次）
```

**Python 解决方案**：
1. **C3 线性化**：确保 MRO 合理
2. **super()**：按 MRO 调用下一个方法
3. **Mixin 模式**：用组合代替继承

**最佳实践**：
- 避免深层次多继承
- 使用 Mixin 提供功能
- 始终使用 `super()`

</details>

---

## 10. 描述符是什么？property 的底层原理？

<details>
<summary>参考答案</summary>

**描述符**：实现了 `__get__`, `__set__`, `__delete__` 任一方法的类。

```python
class Positive:
    """确保值为正数的描述符"""

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"__{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        setattr(obj, self.storage_name, value)

class Product:
    price = Positive()

p = Product()
p.price = 10    # 调用 Positive.__set__
print(p.price)  # 调用 Positive.__get__
```

**property 是描述符**：

```python
# property 本质上是这样实现的
class property:
    def __init__(self, fget=None, fset=None, fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set")
        self.fset(obj, value)
```

</details>

