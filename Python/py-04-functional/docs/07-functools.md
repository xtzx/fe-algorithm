# 07. functools 模块

## 🎯 本节目标

- 掌握 functools 常用函数
- 使用 partial 创建偏函数
- 使用 lru_cache 缓存
- 理解 wraps 和 total_ordering

---

## 📝 functools 概述

`functools` 模块提供了函数式编程的工具函数。

```python
import functools
```

---

## 🔧 partial：偏函数

`partial` 用于**固定函数的部分参数**，创建新函数。

### 基本用法

```python
from functools import partial

# 原函数
def power(base, exponent):
    return base ** exponent

# 固定 base=2，创建新函数
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25 (5**2)
print(cube(3))    # 27 (3**3)
```

### 固定多个参数

```python
def greet(greeting, name, punctuation):
    return f"{greeting}, {name}{punctuation}"

# 固定前两个参数
say_hello = partial(greet, "Hello", punctuation="!")

print(say_hello("Alice"))  # Hello, Alice!
print(say_hello("Bob"))    # Hello, Bob!
```

### 实际应用

```python
# 固定默认参数
def connect(host, port, timeout=10):
    print(f"连接到 {host}:{port}，超时 {timeout}")

# 创建特定环境的连接函数
local_connect = partial(connect, "localhost", 5432)
remote_connect = partial(connect, "api.example.com", 443, timeout=30)

local_connect()      # 连接到 localhost:5432，超时 10
remote_connect()     # 连接到 api.example.com:443，超时 30
```

---

## 💾 lru_cache：缓存装饰器

`lru_cache` 实现**最近最少使用（LRU）缓存**。

### 基本用法

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 快速返回（有缓存）
```

### 参数说明

```python
@lru_cache(maxsize=None)  # 无限制缓存
@lru_cache(maxsize=256)   # 最多缓存 256 个结果
@lru_cache()              # 默认 maxsize=128
```

### 缓存统计

```python
@lru_cache(maxsize=128)
def expensive_function(n):
    return n ** 2

expensive_function(5)
expensive_function(5)  # 从缓存返回

# 查看缓存统计
print(expensive_function.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)
```

### 清除缓存

```python
@lru_cache()
def cached_function(x):
    return x * 2

cached_function(5)
cached_function.cache_clear()  # 清除所有缓存
```

### ⚠️ 注意事项

```python
# ❌ 可变参数不能缓存
@lru_cache()
def bad_function(lst):
    return sum(lst)

bad_function([1, 2, 3])  # TypeError: unhashable type: 'list'

# ✅ 使用不可变参数
@lru_cache()
def good_function(*args):
    return sum(args)

good_function(1, 2, 3)  # ✅
```

---

## 📋 wraps：保留函数元信息

`wraps` 用于装饰器中保留原函数的元信息。

### 不使用 wraps

```python
def timer(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@timer
def my_function():
    """这是文档"""
    pass

print(my_function.__name__)  # wrapper ❌
print(my_function.__doc__)   # None ❌
```

### 使用 wraps

```python
from functools import wraps

def timer(func):
    @wraps(func)  # ✅ 保留元信息
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@timer
def my_function():
    """这是文档"""
    pass

print(my_function.__name__)  # my_function ✅
print(my_function.__doc__)   # 这是文档 ✅
```

---

## 🔢 total_ordering：自动生成比较方法

`total_ordering` 只需实现 `__eq__` 和 `__lt__`，自动生成其他比较方法。

### 不使用 total_ordering

```python
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

    # 还需要实现 __le__, __gt__, __ge__, __ne__
    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    # ... 更多方法
```

### 使用 total_ordering

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)

print(v1 < v2)   # True ✅
print(v1 <= v2)  # True ✅
print(v1 > v2)   # False ✅
print(v1 >= v2)  # False ✅
```

---

## 🔄 reduce：累积操作

`reduce` 将序列归约为单个值。

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda acc, x: acc + x, numbers)
print(total)  # 15

# 等价于
total = sum(numbers)

# 求积
product = reduce(lambda acc, x: acc * x, numbers)
print(product)  # 120

# 带初始值
total = reduce(lambda acc, x: acc + x, numbers, 10)
print(total)  # 25
```

---

## 🎯 其他实用函数

### cmp_to_key：比较函数转 key

```python
from functools import cmp_to_key

def compare(x, y):
    """比较函数：返回 -1, 0, 1"""
    if x < y:
        return -1
    elif x > y:
        return 1
    return 0

numbers = [3, 1, 4, 1, 5]
sorted_numbers = sorted(numbers, key=cmp_to_key(compare))
print(sorted_numbers)  # [1, 1, 3, 4, 5]
```

### singledispatch：单分派泛型函数

```python
from functools import singledispatch

@singledispatch
def process(value):
    """默认处理"""
    return f"处理: {value}"

@process.register
def _(value: int):
    """处理整数"""
    return f"整数: {value}"

@process.register
def _(value: str):
    """处理字符串"""
    return f"字符串: {value}"

print(process(42))      # 整数: 42
print(process("hello")) # 字符串: hello
print(process([1, 2]))  # 处理: [1, 2]
```

---

## ✅ 本节要点

1. `partial` 固定函数参数，创建新函数
2. `lru_cache` 实现缓存，提高性能
3. `wraps` 保留装饰函数的元信息
4. `total_ordering` 自动生成比较方法
5. `reduce` 累积操作
6. `singledispatch` 单分派泛型函数

