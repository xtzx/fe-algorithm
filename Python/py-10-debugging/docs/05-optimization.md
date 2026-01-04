# 05. 性能优化技巧

## 本节目标

- 掌握常见性能问题和解决方案
- 了解数据结构选择对性能的影响
- 建立正确的优化思维

---

## 优化原则

### 1. 不要过早优化

> "Premature optimization is the root of all evil." - Donald Knuth

**正确流程**：
1. 先让代码正确工作
2. 测量找出瓶颈
3. 只优化瓶颈

### 2. 测量，测量，再测量

```python
import time

# 总是先测量
start = time.perf_counter()
result = your_function()
print(f"耗时: {time.perf_counter() - start:.4f}s")
```

---

## 字符串拼接优化

### 问题：使用 += 拼接

```python
# ❌ 慢：每次创建新字符串
s = ""
for i in range(10000):
    s += str(i)  # O(n²)
```

### 解决：使用 join

```python
# ✓ 快：一次性拼接
s = "".join(str(i) for i in range(10000))  # O(n)
```

### 性能对比

```python
import timeit

# += 拼接
def concat_plus():
    s = ""
    for i in range(1000):
        s += str(i)
    return s

# join
def concat_join():
    return "".join(str(i) for i in range(1000))

print(f"+= 拼接: {timeit.timeit(concat_plus, number=1000):.4f}s")
print(f"join: {timeit.timeit(concat_join, number=1000):.4f}s")
```

---

## 列表 vs 生成器

### 问题：不必要的列表

```python
# ❌ 创建中间列表
result = sum([x ** 2 for x in range(1000000)])
```

### 解决：使用生成器

```python
# ✓ 不创建列表
result = sum(x ** 2 for x in range(1000000))
```

### 何时使用生成器

| 场景 | 选择 |
|------|------|
| 只遍历一次 | 生成器 |
| 需要索引访问 | 列表 |
| 需要多次遍历 | 列表 |
| 数据量大 | 生成器 |

---

## 全局查找 vs 局部查找

### 问题：循环中访问全局

```python
import math

# ❌ 慢：每次查找全局
def slow():
    for i in range(10000):
        result = math.sqrt(i)
```

### 解决：缓存到局部

```python
# ✓ 快：使用局部变量
def fast():
    sqrt = math.sqrt  # 缓存到局部
    for i in range(10000):
        result = sqrt(i)
```

### 性能对比

```python
import math
import timeit

def slow():
    total = 0
    for i in range(10000):
        total += math.sqrt(i)
    return total

def fast():
    sqrt = math.sqrt
    total = 0
    for i in range(10000):
        total += sqrt(i)
    return total

print(f"全局: {timeit.timeit(slow, number=1000):.4f}s")
print(f"局部: {timeit.timeit(fast, number=1000):.4f}s")
```

---

## 循环优化

### 减少循环内的工作

```python
# ❌ 每次循环都计算
for item in items:
    if len(items) > threshold:  # 每次都计算 len
        process(item)

# ✓ 提前计算
length = len(items)
for item in items:
    if length > threshold:
        process(item)
```

### 使用列表推导式

```python
# ❌ 慢
result = []
for x in range(1000):
    if x % 2 == 0:
        result.append(x ** 2)

# ✓ 快
result = [x ** 2 for x in range(1000) if x % 2 == 0]
```

### 使用 map/filter

```python
# 列表推导式通常更快
result = [x * 2 for x in data]

# map 有时更快（特别是使用内置函数）
result = list(map(str, data))
```

---

## 数据结构选择

### 查找操作

| 数据结构 | 查找复杂度 |
|----------|-----------|
| list | O(n) |
| set | O(1) |
| dict | O(1) |

```python
# ❌ 列表查找 O(n)
if item in my_list:
    pass

# ✓ 集合查找 O(1)
if item in my_set:
    pass
```

### 队列操作

```python
# ❌ 列表作为队列 O(n)
queue = []
queue.append(item)
queue.pop(0)  # 慢！

# ✓ deque O(1)
from collections import deque
queue = deque()
queue.append(item)
queue.popleft()  # 快
```

### 计数操作

```python
# ❌ 手动计数
counts = {}
for item in items:
    counts[item] = counts.get(item, 0) + 1

# ✓ Counter
from collections import Counter
counts = Counter(items)
```

---

## 缓存

### functools.lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 第一次慢，之后从缓存返回
fibonacci(100)
```

### 手动缓存

```python
_cache = {}

def expensive_function(key):
    if key not in _cache:
        _cache[key] = compute_result(key)
    return _cache[key]
```

---

## 批量操作

### 数据库批量插入

```python
# ❌ 逐条插入
for item in items:
    db.insert(item)

# ✓ 批量插入
db.insert_many(items)
```

### 文件批量写入

```python
# ❌ 逐行写入
for line in lines:
    with open("file.txt", "a") as f:
        f.write(line)

# ✓ 一次写入
with open("file.txt", "w") as f:
    f.writelines(lines)
```

---

## 使用 C 扩展

### NumPy

```python
import numpy as np

# ❌ 纯 Python
def sum_squares_python(n):
    return sum(x ** 2 for x in range(n))

# ✓ NumPy
def sum_squares_numpy(n):
    return np.sum(np.arange(n) ** 2)
```

### 其他选择

- **Cython**: 编译为 C
- **Numba**: JIT 编译
- **PyPy**: JIT 解释器

---

## 并发优化

### I/O 密集型

```python
import asyncio

async def fetch_all(urls):
    # 并发请求
    tasks = [fetch(url) for url in urls]
    return await asyncio.gather(*tasks)
```

### CPU 密集型

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(compute, data))
```

---

## 优化 Checklist

| 检查项 | 解决方案 |
|--------|---------|
| 字符串拼接 | 使用 join |
| 列表查找 | 使用 set/dict |
| 循环内重复计算 | 提前计算 |
| 全局变量频繁访问 | 缓存到局部 |
| 重复计算 | 使用 lru_cache |
| 大数据处理 | 使用生成器 |
| 数值计算 | 使用 NumPy |

---

## 本节要点

1. **不要过早优化**，先测量
2. **字符串拼接** 用 join
3. **生成器** 节省内存
4. **局部变量** 比全局快
5. **选择合适的数据结构**
6. **lru_cache** 缓存结果

