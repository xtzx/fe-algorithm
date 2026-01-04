# 03. 性能分析

## 本节目标

- 使用 cProfile 分析性能
- 了解其他分析工具
- 掌握时间测量最佳实践

---

## cProfile - 标准性能分析器

### 命令行使用

```bash
# 分析整个脚本
python -m cProfile script.py

# 按累计时间排序
python -m cProfile -s cumtime script.py

# 输出到文件
python -m cProfile -o profile.stats script.py
```

### 代码中使用

```python
import cProfile
import pstats
import io

def main():
    # 你的代码
    result = sum(x ** 2 for x in range(100000))
    return result

# 分析
profiler = cProfile.Profile()
profiler.enable()

main()

profiler.disable()

# 打印结果
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(10)  # 前 10 个
```

### 输出解读

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1000    0.150    0.000    0.350    0.000 module.py:10(process)
        │       │        │        │        │
        │       │        │        │        └── 每次调用累计时间
        │       │        │        └── 累计时间（含子函数）
        │       │        └── 每次调用时间
        │       └── 总时间（不含子函数）
        └── 调用次数
```

### 装饰器方式

```python
import cProfile
import functools

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort="cumulative")
        return result
    return wrapper

@profile
def slow_function():
    return sum(x ** 2 for x in range(1000000))
```

---

## 可视化分析结果

### snakeviz

```bash
pip install snakeviz

# 生成统计文件
python -m cProfile -o profile.stats script.py

# 可视化
snakeviz profile.stats
```

### gprof2dot

```bash
pip install gprof2dot

python -m cProfile -o profile.stats script.py
gprof2dot -f pstats profile.stats | dot -Tpng -o profile.png
```

---

## time.perf_counter - 精确计时

```python
import time

def measure_time():
    start = time.perf_counter()

    # 要测量的代码
    result = sum(x ** 2 for x in range(1000000))

    end = time.perf_counter()
    print(f"耗时: {end - start:.4f} 秒")
    return result

# 计时上下文管理器
from contextlib import contextmanager

@contextmanager
def timer(name="代码块"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} 耗时: {end - start:.4f} 秒")

# 使用
with timer("计算"):
    result = sum(range(1000000))
```

---

## timeit - 微基准测试

```python
import timeit

# 测量表达式
time = timeit.timeit("sum(range(1000))", number=10000)
print(f"总耗时: {time:.4f}s")

# 测量函数
def test_func():
    return sum(range(1000))

time = timeit.timeit(test_func, number=10000)
print(f"总耗时: {time:.4f}s")

# 比较两种实现
setup = "data = list(range(1000))"

time1 = timeit.timeit("[x*2 for x in data]", setup=setup, number=10000)
time2 = timeit.timeit("list(map(lambda x: x*2, data))", setup=setup, number=10000)

print(f"列表推导式: {time1:.4f}s")
print(f"map: {time2:.4f}s")
```

---

## line_profiler - 逐行分析

```bash
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
     3                                           @profile
     4                                           def slow_function():
     5         1          1.0      1.0      0.0      total = 0
     6     10001       5000.0      0.5     25.0      for i in range(10000):
     7     10000      15000.0      1.5     75.0          total += i ** 2
     8         1          0.0      0.0      0.0      return total
```

---

## py-spy - 采样分析器

```bash
pip install py-spy
```

```bash
# 分析运行中的进程
py-spy top --pid 12345

# 生成火焰图
py-spy record -o profile.svg -- python script.py

# 不需要修改代码！
```

**优势**：
- 无需修改代码
- 可以附加到运行中的进程
- 低开销

---

## 热点函数识别

### 什么是热点函数

执行时间占比高的函数，优化它们效果最大。

### 识别方法

```python
import cProfile
import pstats

# 运行分析
cProfile.run("main()", "profile.stats")

# 找出热点
stats = pstats.Stats("profile.stats")
stats.sort_stats("cumulative")
stats.print_stats(10)  # 累计时间最长的 10 个函数

stats.sort_stats("tottime")
stats.print_stats(10)  # 自身时间最长的 10 个函数
```

---

## 分析策略

### 80/20 法则

**80% 的时间花在 20% 的代码上**

1. 先找到热点函数
2. 专注优化热点
3. 不要过早优化

### 分析流程

```
1. 测量基准性能
    ↓
2. cProfile 找热点函数
    ↓
3. line_profiler 分析热点函数
    ↓
4. 优化
    ↓
5. 再次测量，验证效果
```

---

## 与 JS 工具对比

| 功能 | Python | JavaScript |
|------|--------|------------|
| 函数级分析 | cProfile | Chrome DevTools |
| 行级分析 | line_profiler | - |
| 采样分析 | py-spy | Chrome Sampling |
| 可视化 | snakeviz | Chrome DevTools |
| 微基准 | timeit | benchmark.js |

---

## 本节要点

1. **cProfile** 标准函数级分析器
2. **cumtime** 累计时间，**tottime** 自身时间
3. **timeit** 微基准测试
4. **line_profiler** 逐行分析
5. **py-spy** 无侵入采样分析
6. **80/20** 法则：专注热点

