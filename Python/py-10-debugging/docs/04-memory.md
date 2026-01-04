# 04. 内存分析

## 本节目标

- 使用 tracemalloc 分析内存
- 定位内存泄漏
- 了解内存优化技巧

---

## tracemalloc - 标准内存分析器

### 基本使用

```python
import tracemalloc

# 开始追踪
tracemalloc.start()

# 运行代码
data = [x ** 2 for x in range(100000)]

# 获取快照
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("[ 内存使用 Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

# 停止追踪
tracemalloc.stop()
```

### 输出示例

```
[ 内存使用 Top 10 ]
script.py:5: size=3912 KiB, count=100000, average=40 B
script.py:8: size=448 KiB, count=10000, average=46 B
...
```

---

## 比较内存快照

```python
import tracemalloc

tracemalloc.start()

# 第一个快照
snapshot1 = tracemalloc.take_snapshot()

# 运行可能泄漏的代码
leaked_data = []
for i in range(1000):
    leaked_data.append([0] * 1000)

# 第二个快照
snapshot2 = tracemalloc.take_snapshot()

# 比较差异
diff = snapshot2.compare_to(snapshot1, "lineno")

print("[ 内存增长 ]")
for stat in diff[:10]:
    print(stat)
```

---

## 追踪内存峰值

```python
import tracemalloc

tracemalloc.start()

# 运行代码
data = [x ** 2 for x in range(100000)]

# 获取当前和峰值
current, peak = tracemalloc.get_traced_memory()

print(f"当前内存: {current / 1024 / 1024:.2f} MB")
print(f"峰值内存: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

---

## memory_profiler - 逐行内存分析

```bash
pip install memory_profiler
```

```python
# script.py
from memory_profiler import profile

@profile
def memory_hungry():
    a = [1] * 1000000
    b = [2] * 2000000
    del b
    return a

memory_hungry()
```

```bash
python -m memory_profiler script.py
```

输出：
```
Line #    Mem usage    Increment   Line Contents
     4     38.0 MiB     38.0 MiB   @profile
     5                             def memory_hungry():
     6     45.6 MiB      7.6 MiB       a = [1] * 1000000
     7     61.0 MiB     15.3 MiB       b = [2] * 2000000
     8     45.6 MiB    -15.3 MiB       del b
     9     45.6 MiB      0.0 MiB       return a
```

---

## 定位内存泄漏

### 常见泄漏原因

1. **循环引用**
2. **全局变量累积**
3. **闭包捕获**
4. **缓存无限增长**
5. **未关闭的资源**

### 检测循环引用

```python
import gc

# 启用调试
gc.set_debug(gc.DEBUG_LEAK)

# 强制收集
gc.collect()

# 查看不可回收对象
print(f"不可回收对象: {len(gc.garbage)}")
```

### 使用弱引用

```python
import weakref

class Node:
    def __init__(self):
        self.ref = None

# 强引用导致循环
a = Node()
b = Node()
a.ref = b
b.ref = a  # 循环引用！

# 使用弱引用
a = Node()
b = Node()
a.ref = b
b.ref = weakref.ref(a)  # 弱引用
```

---

## objgraph - 对象图分析

```bash
pip install objgraph
```

```python
import objgraph

# 查看最常见的对象类型
objgraph.show_most_common_types(limit=10)

# 查看增长最快的类型
objgraph.show_growth()

# 查找对象的引用
obj = SomeClass()
objgraph.show_backrefs(obj, max_depth=3, filename="refs.png")
```

---

## 内存优化技巧

### 使用 `__slots__`

```python
class PointWithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointWithSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# 节省约 40% 内存
import sys
print(sys.getsizeof(PointWithoutSlots(1, 2)))  # 更大
print(sys.getsizeof(PointWithSlots(1, 2)))     # 更小
```

### 使用生成器

```python
# 列表：立即占用内存
data = [x ** 2 for x in range(1000000)]  # ~8MB

# 生成器：按需生成
data = (x ** 2 for x in range(1000000))  # ~100B
```

### 使用 array

```python
import array

# list 存储 Python 对象
lst = [1, 2, 3, 4, 5]

# array 存储原始数据
arr = array.array('i', [1, 2, 3, 4, 5])  # 更小
```

### 使用 numpy

```python
import numpy as np

# Python list
py_list = [i for i in range(1000000)]

# NumPy array
np_array = np.arange(1000000)

# NumPy 更节省内存且更快
```

---

## 监控生产环境内存

### 使用 psutil

```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"当前内存: {get_memory_usage():.2f} MB")
```

### 定期监控

```python
import threading
import time
import psutil
import os

def memory_monitor(interval=60):
    process = psutil.Process(os.getpid())
    while True:
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"内存使用: {mem_mb:.2f} MB")
        time.sleep(interval)

# 后台监控
monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
monitor_thread.start()
```

---

## 内存泄漏排查流程

```
1. 观察：内存持续增长
    ↓
2. tracemalloc 定位增长位置
    ↓
3. 比较快照找出差异
    ↓
4. objgraph 分析引用链
    ↓
5. 修复：释放引用/使用弱引用
    ↓
6. 验证：确认内存稳定
```

---

## 本节要点

1. **tracemalloc** 标准内存分析器
2. **比较快照** 找出内存增长
3. **memory_profiler** 逐行分析
4. **循环引用** 是常见泄漏原因
5. **弱引用** 避免循环引用
6. **`__slots__`** 和生成器节省内存

