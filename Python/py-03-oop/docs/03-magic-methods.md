# 03. 魔法方法

## 🎯 本节目标

- 理解常用魔法方法
- 实现自定义类的运算符重载
- 掌握容器协议

---

## 📝 什么是魔法方法

魔法方法（Dunder Methods）是以双下划线开头和结尾的特殊方法，用于定义对象的行为。

```python
class Vector:
    def __init__(self, x, y):  # 构造
        self.x = x
        self.y = y

    def __repr__(self):        # 表示
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):  # + 运算符
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)  # Vector(4, 6)
```

---

## 🏗️ 构造与析构

```python
class Resource:
    def __new__(cls, *args, **kwargs):
        """创建实例（在 __init__ 之前）"""
        print("__new__ called")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        """初始化实例"""
        print("__init__ called")
        self.name = name

    def __del__(self):
        """析构（垃圾回收时调用）"""
        print(f"__del__ called for {self.name}")

r = Resource("test")
# __new__ called
# __init__ called
del r
# __del__ called for test
```

### `__new__` vs `__init__`

| 方法 | 作用 | 返回值 |
|------|------|--------|
| `__new__` | 创建实例 | 必须返回实例 |
| `__init__` | 初始化实例 | None |

**`__new__` 的用途**：
- 单例模式
- 不可变对象（如自定义 str/int）
- 元类

---

## 📜 字符串表示

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        """用户友好的字符串（print 时使用）"""
        return f"{self.name}, {self.age} years old"

    def __repr__(self):
        """开发者友好的字符串（调试时使用）"""
        return f"Person(name={self.name!r}, age={self.age})"

p = Person("Alice", 25)
print(str(p))   # Alice, 25 years old
print(repr(p))  # Person(name='Alice', age=25)
print(p)        # Alice, 25 years old（调用 __str__）
```

### `__str__` vs `__repr__`

| 方法 | 用途 | 调用时机 |
|------|------|---------|
| `__str__` | 用户友好 | `str()`, `print()` |
| `__repr__` | 开发者友好 | `repr()`, 交互式环境 |

> 如果只实现一个，实现 `__repr__`

---

## ⚖️ 比较方法

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        """=="""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        """!=（通常自动从 __eq__ 推导）"""
        return not self.__eq__(other)

    def __lt__(self, other):
        """<"""
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __le__(self, other):
        """<="""
        return self == other or self < other

    def __gt__(self, other):
        """>"""
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) > (other.x, other.y)

    def __ge__(self, other):
        """>="""
        return self == other or self > other

    def __hash__(self):
        """哈希值（实现 __eq__ 后需要实现）"""
        return hash((self.x, self.y))

p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(2, 3)

print(p1 == p2)  # True
print(p1 < p3)   # True
print({p1, p2})  # 只有一个元素（因为相等）
```

### 使用 functools.total_ordering

```python
from functools import total_ordering

@total_ordering
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    # 其他比较方法自动生成！
```

---

## ➕ 算术运算符

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """self + other"""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """self - other"""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """self * scalar"""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        """scalar * self（反向）"""
        return self.__mul__(scalar)

    def __neg__(self):
        """-self"""
        return Vector(-self.x, -self.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v = Vector(1, 2)
print(v + Vector(3, 4))  # Vector(4, 6)
print(v * 3)             # Vector(3, 6)
print(3 * v)             # Vector(3, 6)
print(-v)                # Vector(-1, -2)
```

---

## 📦 容器协议

```python
class MyList:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        """len(obj)"""
        return len(self._items)

    def __getitem__(self, index):
        """obj[index]"""
        return self._items[index]

    def __setitem__(self, index, value):
        """obj[index] = value"""
        self._items[index] = value

    def __delitem__(self, index):
        """del obj[index]"""
        del self._items[index]

    def __contains__(self, item):
        """item in obj"""
        return item in self._items

    def __iter__(self):
        """for item in obj"""
        return iter(self._items)

lst = MyList([1, 2, 3])
print(len(lst))    # 3
print(lst[0])      # 1
print(2 in lst)    # True
for item in lst:
    print(item)
```

---

## 📞 可调用对象

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        """使对象可调用"""
        return self.n + x

add_5 = Adder(5)
print(add_5(10))  # 15
print(callable(add_5))  # True
```

---

## 🚪 上下文管理器

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """进入 with 块"""
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块"""
        if self.file:
            self.file.close()
        return False  # 不抑制异常

with FileManager("test.txt", "w") as f:
    f.write("Hello")
# 自动关闭
```

---

## ✅ 本节要点

1. `__repr__` 优先于 `__str__`
2. 实现 `__eq__` 后记得实现 `__hash__`
3. 容器协议：`__len__`, `__getitem__`, `__iter__`
4. `__call__` 使对象可调用
5. `__enter__` / `__exit__` 实现上下文管理器

