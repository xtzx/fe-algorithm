# 高级内存优化

> 内存分析、泄漏检测、__slots__、对象池与大数据处理

## 内存分析工具

### memory_profiler

```bash
pip install memory_profiler
```

```python
# script.py
from memory_profiler import profile

@profile
def memory_heavy():
    a = [i ** 2 for i in range(1_000_000)]
    b = [i ** 3 for i in range(1_000_000)]
    del a
    return b

if __name__ == '__main__':
    memory_heavy()
```

```bash
python -m memory_profiler script.py
```

输出：

```
Line #    Mem usage    Increment   Line Contents
================================================
     4     50.0 MiB     50.0 MiB   @profile
     5                             def memory_heavy():
     6     88.5 MiB     38.5 MiB       a = [i ** 2 for i in range(1_000_000)]
     7    127.0 MiB     38.5 MiB       b = [i ** 3 for i in range(1_000_000)]
     8     88.5 MiB    -38.5 MiB       del a
     9     88.5 MiB      0.0 MiB       return b
```

### tracemalloc（标准库）

```python
import tracemalloc

tracemalloc.start()

# 你的代码
data = [i ** 2 for i in range(1_000_000)]

# 获取快照
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
```

### objgraph（对象引用图）

```bash
pip install objgraph
```

```python
import objgraph

class MyClass:
    pass

obj = MyClass()
obj.self_ref = obj  # 循环引用

# 找出最多的对象类型
objgraph.show_most_common_types(limit=10)

# 找出增长最快的类型
objgraph.show_growth()

# 显示引用链
objgraph.show_backrefs([obj], filename='refs.png')
```

---

## 内存泄漏检测

### 常见泄漏模式

```python
# 1. 全局列表无限增长
cache = []

def process(data):
    result = expensive_compute(data)
    cache.append(result)  # ❌ 永远不清理
    return result

# ✅ 使用 LRU 缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def process(data):
    return expensive_compute(data)
```

```python
# 2. 循环引用
class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # 循环引用

# ✅ 使用弱引用
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self._parent = None
        self.children = []

    @property
    def parent(self):
        return self._parent() if self._parent else None

    @parent.setter
    def parent(self, node):
        self._parent = weakref.ref(node) if node else None
```

```python
# 3. 闭包捕获
def create_handlers():
    handlers = []
    large_data = [0] * 10_000_000  # 大对象

    for i in range(100):
        def handler():
            return large_data[i]  # ❌ 捕获了 large_data
        handlers.append(handler)

    return handlers

# ✅ 只捕获需要的值
def create_handlers():
    handlers = []
    large_data = [0] * 10_000_000

    for i in range(100):
        value = large_data[i]  # 提取值
        def handler(v=value):  # 只捕获值
            return v
        handlers.append(handler)

    return handlers
```

### 检测泄漏

```python
import gc
import tracemalloc

def detect_leak():
    tracemalloc.start()

    # 第一次快照
    snapshot1 = tracemalloc.take_snapshot()

    # 执行可能泄漏的代码
    for _ in range(100):
        do_something()

    gc.collect()

    # 第二次快照
    snapshot2 = tracemalloc.take_snapshot()

    # 比较
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ 内存增长 Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
```

---

## __slots__ 优化

### 基础用法

```python
# 普通类
class NormalPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 使用 __slots__
class SlottedPoint:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### 内存对比

```python
import sys

normal = NormalPoint(1, 2)
slotted = SlottedPoint(1, 2)

print(f"Normal: {sys.getsizeof(normal)} + {sys.getsizeof(normal.__dict__)} bytes")
print(f"Slotted: {sys.getsizeof(slotted)} bytes")

# 典型输出:
# Normal: 48 + 104 bytes = 152 bytes
# Slotted: 48 bytes
```

### 大量对象场景

```python
import tracemalloc

tracemalloc.start()

# 创建 100 万个对象
normal_points = [NormalPoint(i, i) for i in range(1_000_000)]
current, peak = tracemalloc.get_traced_memory()
print(f"Normal: {current / 1024 / 1024:.1f} MB")

tracemalloc.reset_peak()

slotted_points = [SlottedPoint(i, i) for i in range(1_000_000)]
current, peak = tracemalloc.get_traced_memory()
print(f"Slotted: {current / 1024 / 1024:.1f} MB")

# 典型输出:
# Normal: 200+ MB
# Slotted: 80 MB
```

### __slots__ 注意事项

```python
# 1. 继承时需要重新声明
class Base:
    __slots__ = ['x']

class Derived(Base):
    __slots__ = ['y']  # 只声明新属性

# 2. 允许动态属性
class Flexible:
    __slots__ = ['x', '__dict__']  # 允许添加新属性

# 3. 不能使用默认值
class Bad:
    __slots__ = ['x']
    x = 10  # ❌ 错误

class Good:
    __slots__ = ['x']

    def __init__(self):
        self.x = 10  # ✅ 正确
```

---

## 对象池

### 基础对象池

```python
from typing import TypeVar, Generic
from collections import deque

T = TypeVar('T')

class ObjectPool(Generic[T]):
    """简单对象池"""

    def __init__(self, factory, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self._pool: deque = deque()

    def acquire(self) -> T:
        """获取对象"""
        if self._pool:
            return self._pool.pop()
        return self.factory()

    def release(self, obj: T):
        """归还对象"""
        if len(self._pool) < self.max_size:
            self._pool.append(obj)

# 使用
class ExpensiveObject:
    def __init__(self):
        self.data = [0] * 10000

    def reset(self):
        self.data = [0] * 10000

pool = ObjectPool(ExpensiveObject, max_size=50)

# 获取
obj = pool.acquire()
# 使用 obj...

# 归还（重置后）
obj.reset()
pool.release(obj)
```

### 上下文管理器版本

```python
from contextlib import contextmanager

class ManagedPool:
    def __init__(self, factory, max_size=100):
        self.factory = factory
        self.max_size = max_size
        self._pool = []

    @contextmanager
    def acquire(self):
        obj = self._pool.pop() if self._pool else self.factory()
        try:
            yield obj
        finally:
            if len(self._pool) < self.max_size:
                self._pool.append(obj)

# 使用
pool = ManagedPool(list, max_size=10)

with pool.acquire() as buffer:
    buffer.clear()
    buffer.extend([1, 2, 3])
    # 自动归还
```

---

## 大数据处理

### 分块读取

```python
def process_large_file(filename, chunk_size=1024*1024):
    """分块读取大文件"""
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            process_chunk(chunk)

# 使用生成器
def read_chunks(filename, chunk_size=1024*1024):
    with open(filename, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk

for chunk in read_chunks('large_file.bin'):
    process_chunk(chunk)
```

### Pandas 分块

```python
import pandas as pd

# ❌ 一次性读取（可能 OOM）
# df = pd.read_csv('huge.csv')

# ✅ 分块读取
chunk_iter = pd.read_csv('huge.csv', chunksize=100_000)

results = []
for chunk in chunk_iter:
    # 处理每个块
    result = process_chunk(chunk)
    results.append(result)

# 合并结果
final = pd.concat(results)
```

### 使用 Dask

```python
import dask.dataframe as dd

# 延迟加载，自动分块
df = dd.read_csv('huge_*.csv')

# 操作（延迟执行）
result = df.groupby('category').sum()

# 执行并获取结果
final = result.compute()
```

### 内存映射

```python
import numpy as np

# 创建大数组（保存到磁盘）
large_array = np.memmap(
    'large_array.dat',
    dtype='float64',
    mode='w+',
    shape=(10_000_000,)
)

# 像普通数组一样操作
large_array[0] = 1.0
large_array[1:100] = np.arange(99)

# 刷新到磁盘
large_array.flush()

# 重新打开（只读）
loaded = np.memmap('large_array.dat', dtype='float64', mode='r')
```

---

## 字符串优化

### 字符串驻留（Interning）

```python
import sys

# Python 自动驻留短字符串
a = "hello"
b = "hello"
print(a is b)  # True

# 长字符串不自动驻留
a = "hello world " * 100
b = "hello world " * 100
print(a is b)  # False

# 手动驻留
a = sys.intern("hello world " * 100)
b = sys.intern("hello world " * 100)
print(a is b)  # True
```

### 使用 bytes 代替 str

```python
# 处理 ASCII 数据时，bytes 更高效
text_str = "hello" * 1_000_000
text_bytes = b"hello" * 1_000_000

print(f"str: {sys.getsizeof(text_str)} bytes")
print(f"bytes: {sys.getsizeof(text_bytes)} bytes")
```

---

## 数据结构替代

### array vs list

```python
import array
import sys

# 列表（对象引用）
int_list = [0] * 1_000_000
print(f"list: {sys.getsizeof(int_list)} bytes")

# array（紧凑存储）
int_array = array.array('l', [0] * 1_000_000)
print(f"array: {sys.getsizeof(int_array)} bytes")

# 典型输出:
# list: 8,000,056 bytes
# array: 8,000,064 bytes（但元素本身更紧凑）
```

### 使用 struct

```python
import struct

# 打包二进制数据
data = struct.pack('3i', 1, 2, 3)  # 3 个 int
print(len(data))  # 12 bytes

# 解包
values = struct.unpack('3i', data)
print(values)  # (1, 2, 3)
```

### NumPy 数据类型

```python
import numpy as np

# 选择合适的数据类型
arr_float64 = np.zeros(1_000_000, dtype=np.float64)  # 8 MB
arr_float32 = np.zeros(1_000_000, dtype=np.float32)  # 4 MB
arr_int8 = np.zeros(1_000_000, dtype=np.int8)        # 1 MB

# 对于 0-255 的整数，使用 uint8
image = np.zeros((1920, 1080, 3), dtype=np.uint8)  # 6 MB
# 而不是默认的 float64: 50 MB
```

---

## 垃圾回收调优

### 手动触发 GC

```python
import gc

# 显式回收
gc.collect()

# 获取 GC 统计
print(gc.get_stats())
```

### 禁用 GC（谨慎）

```python
import gc

# 对于短期大量创建对象的场景
gc.disable()
try:
    # 创建大量临时对象
    result = process_data()
finally:
    gc.enable()
    gc.collect()
```

### 调整 GC 阈值

```python
import gc

# 获取当前阈值
print(gc.get_threshold())  # (700, 10, 10)

# 调整阈值（减少 GC 频率）
gc.set_threshold(1000, 15, 15)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 滥用 __slots__ | 增加代码复杂度 | 只在大量对象时使用 |
| 忽略 del | del 只删除引用 | 循环引用需要 gc.collect() |
| 缓存无限增长 | 内存泄漏 | 使用 lru_cache 或设置上限 |
| 大文件一次读取 | OOM | 分块读取或内存映射 |

---

## 小结

1. **分析工具**：memory_profiler、tracemalloc、objgraph
2. **__slots__**：大量对象时节省 40-60% 内存
3. **对象池**：复用昂贵对象，减少 GC 压力
4. **大数据**：分块处理、Dask、内存映射
5. **数据类型**：选择合适的 NumPy dtype
6. **避免泄漏**：弱引用处理循环引用

