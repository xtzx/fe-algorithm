# 06. 迭代器（Iterator）

## 🎯 本节目标

- 理解迭代器协议
- 实现自定义迭代器
- 掌握 iter() 和 next()
- 了解 itertools 模块

---

## 📝 什么是迭代器

迭代器（Iterator）是实现了迭代器协议的对象，可以逐个访问元素。

### 迭代器协议

迭代器必须实现两个方法：
1. `__iter__()`：返回迭代器本身
2. `__next__()`：返回下一个值，没有时抛出 `StopIteration`

```python
class Countdown:
    """倒计时迭代器"""
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# 使用
for i in Countdown(5):
    print(i)
# 5
# 4
# 3
# 2
# 1
```

---

## 🔄 iter() 和 next()

### iter()：创建迭代器

```python
# 从可迭代对象创建迭代器
numbers = [1, 2, 3]
it = iter(numbers)

print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
# print(next(it))  # StopIteration
```

### next()：获取下一个值

```python
numbers = [1, 2, 3]
it = iter(numbers)

# 方式 1：直接调用
value = next(it)

# 方式 2：提供默认值
value = next(it, None)  # 没有值时返回 None
```

### 手动迭代

```python
numbers = [1, 2, 3]
it = iter(numbers)

while True:
    try:
        value = next(it)
        print(value)
    except StopIteration:
        break
```

---

## 🎨 可迭代对象 vs 迭代器

### 可迭代对象（Iterable）

实现了 `__iter__()` 方法的对象。

```python
class MyList:
    """可迭代对象"""
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)  # 返回迭代器

my_list = MyList([1, 2, 3])
for item in my_list:
    print(item)
```

### 迭代器（Iterator）

实现了 `__iter__()` 和 `__next__()` 的对象。

```python
class MyIterator:
    """迭代器"""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        value = self.items[self.index]
        self.index += 1
        return value
```

### 关系

- **可迭代对象**：可以用 `for` 循环
- **迭代器**：可以调用 `next()`
- **所有迭代器都是可迭代的**
- **不是所有可迭代对象都是迭代器**

```python
# 列表是可迭代对象，但不是迭代器
numbers = [1, 2, 3]
print(iter(numbers))  # <list_iterator object>

# 生成器是迭代器
gen = (x for x in range(3))
print(iter(gen) is gen)  # True
```

---

## 🛠️ 自定义迭代器类

### 简单迭代器

```python
class Range:
    """自定义 range"""
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return RangeIterator(self.start, self.stop, self.step)

class RangeIterator:
    def __init__(self, start, stop, step):
        self.current = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        value = self.current
        self.current += self.step
        return value

for i in Range(0, 5):
    print(i)  # 0, 1, 2, 3, 4
```

### 无限迭代器

```python
class InfiniteCounter:
    """无限计数器"""
    def __init__(self, start=0, step=1):
        self.current = start
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        value = self.current
        self.current += self.step
        return value

counter = InfiniteCounter()
for i, value in enumerate(counter):
    if i >= 10:
        break
    print(value)
```

---

## 📦 容器协议

实现 `__getitem__()` 的对象也可以迭代。

```python
class MySequence:
    """通过 __getitem__ 实现可迭代"""
    def __init__(self, items):
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

seq = MySequence([1, 2, 3])
for item in seq:
    print(item)  # 1, 2, 3
```

---

## 🔍 检查可迭代性

```python
from collections.abc import Iterable, Iterator

# 检查是否可迭代
print(isinstance([1, 2, 3], Iterable))  # True
print(isinstance("hello", Iterable))    # True
print(isinstance(123, Iterable))        # False

# 检查是否是迭代器
print(isinstance([1, 2, 3], Iterator))  # False
print(isinstance(iter([1, 2, 3]), Iterator))  # True
print(isinstance((x for x in range(3)), Iterator))  # True
```

---

## 🎯 实际应用

### 1. 文件读取器

```python
class FileReader:
    """文件读取迭代器"""
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        self.file = open(self.filename)
        return self

    def __next__(self):
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()

for line in FileReader("data.txt"):
    print(line)
```

### 2. 分块迭代器

```python
class ChunkIterator:
    """分块迭代器"""
    def __init__(self, iterable, chunk_size):
        self.iterator = iter(iterable)
        self.chunk_size = chunk_size

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        for _ in range(self.chunk_size):
            try:
                chunk.append(next(self.iterator))
            except StopIteration:
                if chunk:
                    return chunk
                raise
        return chunk

data = range(10)
for chunk in ChunkIterator(data, 3):
    print(chunk)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]
```

---

## 🔗 迭代器组合

### 链式迭代器

```python
class ChainIterator:
    """链式迭代器"""
    def __init__(self, *iterables):
        self.iterables = iterables
        self.current = None
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current is None:
                if self.index >= len(self.iterables):
                    raise StopIteration
                self.current = iter(self.iterables[self.index])
                self.index += 1

            try:
                return next(self.current)
            except StopIteration:
                self.current = None

for value in ChainIterator([1, 2], [3, 4], [5]):
    print(value)
# 1, 2, 3, 4, 5
```

---

## ⚠️ 常见陷阱

### 1. 迭代器只能使用一次

```python
it = iter([1, 2, 3])
list(it)  # [1, 2, 3]
list(it)  # []（已耗尽）
```

### 2. 修改迭代中的集合

```python
# ❌ 危险
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)  # 可能出错

# ✅ 安全：遍历副本
for n in numbers[:]:
    if n % 2 == 0:
        numbers.remove(n)
```

---

## ✅ 本节要点

1. 迭代器实现 `__iter__()` 和 `__next__()`
2. `iter()` 创建迭代器，`next()` 获取下一个值
3. 可迭代对象可以用 `for` 循环
4. 迭代器只能使用一次
5. 生成器是迭代器的特殊形式
6. 自定义迭代器类需要实现迭代器协议

