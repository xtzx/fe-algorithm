# 07. 性能工具

## 本节目标

- 掌握 Python 性能分析工具
- 学会计时和性能测量
- 了解内存分析方法

---

## time.perf_counter - 精确计时

### 基本用法

```python
import time

start = time.perf_counter()

# 执行代码
result = sum(i ** 2 for i in range(1000000))

end = time.perf_counter()
print(f"耗时: {end - start:.4f} 秒")
```

### 计时装饰器

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

slow_function()  # slow_function 耗时: 1.0012s
```

### 计时上下文管理器

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name="代码块"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} 耗时: {end - start:.4f}s")

with timer("计算"):
    result = sum(range(1000000))
```

---

## timeit - 微基准测试

### 命令行使用

```bash
# 测量表达式执行时间
python3 -m timeit "sum(range(1000))"
# 10000 loops, best of 5: 20.3 usec per loop

# 多行代码
python3 -m timeit -s "import math" "math.sqrt(2)"
```

### 代码中使用

```python
import timeit

# 测量代码片段
time = timeit.timeit('sum(range(1000))', number=10000)
print(f"总耗时: {time:.4f}s")

# 测量函数
def test_func():
    return sum(range(1000))

time = timeit.timeit(test_func, number=10000)
print(f"总耗时: {time:.4f}s")
```

### 比较不同实现

```python
import timeit

# 列表推导式 vs map
setup = "data = list(range(1000))"

time1 = timeit.timeit(
    "[x * 2 for x in data]",
    setup=setup,
    number=10000
)

time2 = timeit.timeit(
    "list(map(lambda x: x * 2, data))",
    setup=setup,
    number=10000
)

print(f"列表推导式: {time1:.4f}s")
print(f"map: {time2:.4f}s")
```

---

## cProfile - 性能分析

### 命令行使用

```bash
python3 -m cProfile script.py
python3 -m cProfile -s cumtime script.py  # 按累计时间排序
python3 -m cProfile -o profile.stats script.py  # 输出到文件
```

### 代码中使用

```python
import cProfile
import pstats

def main():
    # 你的代码
    result = sum(i ** 2 for i in range(100000))
    return result

# 分析
profiler = cProfile.Profile()
profiler.enable()

main()

profiler.disable()

# 打印结果
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 前 10 个
```

### 输出解读

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.050    0.001    0.150    0.002 module.py:10(func)
        │       │        │        │        │
        │       │        │        │        └── 每次调用累计时间
        │       │        │        └── 累计时间（含子函数）
        │       │        └── 每次调用时间
        │       └── 总时间（不含子函数）
        └── 调用次数
```

### 使用装饰器

```python
import cProfile
import pstats
import io
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        print(s.getvalue())

        return result
    return wrapper

@profile
def my_function():
    # 代码
    pass
```

---

## line_profiler - 逐行分析

```bash
# 安装
pip install line_profiler
```

```python
# script.py
@profile  # line_profiler 提供的装饰器
def slow_function():
    total = 0
    for i in range(10000):
        total += i ** 2
    return total

slow_function()
```

```bash
kernprof -l -v script.py
```

输出：
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
     2                                           def slow_function():
     3         1          1.0      1.0      0.0      total = 0
     4     10001       5000.0      0.5     25.0      for i in range(10000):
     5     10000      15000.0      1.5     75.0          total += i ** 2
     6         1          0.0      0.0      0.0      return total
```

---

## tracemalloc - 内存分析

### 基本用法

```python
import tracemalloc

tracemalloc.start()

# 运行代码
data = [i ** 2 for i in range(100000)]

# 获取快照
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ 内存使用 Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

# 当前和峰值
current, peak = tracemalloc.get_traced_memory()
print(f"\n当前: {current / 1024:.2f} KB")
print(f"峰值: {peak / 1024:.2f} KB")

tracemalloc.stop()
```

### 追踪内存增长

```python
import tracemalloc

tracemalloc.start()

# 第一个快照
snapshot1 = tracemalloc.take_snapshot()

# 执行可能泄漏的代码
leaked_data = []
for i in range(1000):
    leaked_data.append([0] * 1000)

# 第二个快照
snapshot2 = tracemalloc.take_snapshot()

# 比较差异
diff = snapshot2.compare_to(snapshot1, 'lineno')

print("[ 内存增长 ]")
for stat in diff[:10]:
    print(stat)
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
python3 -m memory_profiler script.py
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

## 优化策略

### 常见优化点

| 问题 | 解决方案 |
|------|----------|
| 循环慢 | 使用列表推导式、numpy |
| 字符串拼接慢 | 使用 `''.join()` |
| 全局变量访问慢 | 使用局部变量 |
| 函数调用开销 | 内联简单函数 |
| 属性访问慢 | 缓存到局部变量 |

### 示例：字符串拼接

```python
import timeit

# 慢：字符串拼接
def slow():
    s = ""
    for i in range(10000):
        s += str(i)
    return s

# 快：join
def fast():
    return "".join(str(i) for i in range(10000))

print(f"拼接: {timeit.timeit(slow, number=100):.4f}s")
print(f"join: {timeit.timeit(fast, number=100):.4f}s")
```

### 示例：缓存局部变量

```python
import timeit

# 慢：重复属性访问
def slow():
    import math
    total = 0
    for i in range(10000):
        total += math.sqrt(i)
    return total

# 快：缓存到局部变量
def fast():
    from math import sqrt
    total = 0
    for i in range(10000):
        total += sqrt(i)
    return total

print(f"属性访问: {timeit.timeit(slow, number=1000):.4f}s")
print(f"局部变量: {timeit.timeit(fast, number=1000):.4f}s")
```

---

## 工具选择指南

| 需求 | 工具 |
|------|------|
| 简单计时 | `time.perf_counter()` |
| 微基准测试 | `timeit` |
| 函数级分析 | `cProfile` |
| 逐行分析 | `line_profiler` |
| 内存总量 | `tracemalloc` |
| 逐行内存 | `memory_profiler` |
| 可视化 | `snakeviz`、`py-spy` |

---

## 本节要点

1. **time.perf_counter()**: 高精度计时
2. **timeit**: 微基准测试，多次运行取平均
3. **cProfile**: 函数级 CPU 分析
4. **line_profiler**: 逐行 CPU 分析
5. **tracemalloc**: 内存追踪和泄漏检测
6. **memory_profiler**: 逐行内存分析
7. **优化策略**: 局部变量、列表推导式、join

