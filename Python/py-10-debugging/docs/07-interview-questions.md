# 07. 面试题

## 1. 如何调试 Python 程序？

**答案**：

### 方法一：pdb / breakpoint()

```python
def buggy_function():
    x = 1
    breakpoint()  # Python 3.7+
    y = x / 0
```

### 方法二：IDE 调试

- VS Code：设置断点，F5 启动
- PyCharm：同样方式

### 方法三：print 调试

```python
print(f"DEBUG: x = {x}")  # 简单但不推荐生产环境
```

### 方法四：logging

```python
import logging
logging.debug(f"x = {x}")
```

### 常用 pdb 命令

- `n`: 下一行
- `s`: 进入函数
- `c`: 继续执行
- `p expr`: 打印表达式
- `l`: 显示代码

---

## 2. logging 和 print 的区别？

**答案**：

| 特性 | logging | print |
|------|---------|-------|
| 级别控制 | ✓ DEBUG/INFO/WARNING/ERROR | ✗ |
| 输出目标 | 文件/控制台/网络等 | 只有 stdout |
| 格式化 | 时间/文件/行号等 | 基础 |
| 可禁用 | ✓ | ✗ |
| 线程安全 | ✓ | ✓ |
| 生产环境 | ✓ | ✗ |

**结论**：生产环境必须使用 logging。

---

## 3. 如何定位 Python 的性能瓶颈？

**答案**：

### 步骤一：cProfile 找热点函数

```bash
python -m cProfile -s cumtime script.py
```

### 步骤二：line_profiler 分析热点

```python
@profile
def hot_function():
    # 代码
```

```bash
kernprof -l -v script.py
```

### 步骤三：timeit 微基准测试

```python
import timeit
timeit.timeit("your_code()", number=1000)
```

### 关键指标

- **cumtime**: 累计时间（含子函数）
- **tottime**: 自身时间
- **ncalls**: 调用次数

---

## 4. 如何优化 Python 循环？

**答案**：

### 1. 使用列表推导式

```python
# 慢
result = []
for x in range(1000):
    result.append(x * 2)

# 快
result = [x * 2 for x in range(1000)]
```

### 2. 减少循环内的工作

```python
# 慢
for item in items:
    if len(items) > 10:  # 每次都计算
        process(item)

# 快
length = len(items)
for item in items:
    if length > 10:
        process(item)
```

### 3. 缓存到局部变量

```python
# 慢
for i in range(10000):
    result = math.sqrt(i)

# 快
sqrt = math.sqrt
for i in range(10000):
    result = sqrt(i)
```

### 4. 使用 NumPy

```python
import numpy as np
result = np.sqrt(np.arange(10000))
```

---

## 5. Python 的内存泄漏怎么排查？

**答案**：

### 步骤一：tracemalloc 找增长

```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# 运行可疑代码

snapshot2 = tracemalloc.take_snapshot()
diff = snapshot2.compare_to(snapshot1, "lineno")
for stat in diff[:10]:
    print(stat)
```

### 步骤二：objgraph 分析引用

```python
import objgraph
objgraph.show_growth()
objgraph.show_backrefs(obj)
```

### 常见原因

1. **循环引用** + `__del__`
2. **全局变量累积**
3. **缓存无限增长**
4. **闭包捕获**
5. **未关闭的资源**

---

## 6. 什么是热点函数？

**答案**：

热点函数是执行时间占比高的函数。

**80/20 法则**：80% 的时间花在 20% 的代码上。

**识别方法**：

```python
import cProfile
import pstats

cProfile.run("main()", "stats")
stats = pstats.Stats("stats")
stats.sort_stats("cumulative")
stats.print_stats(10)  # 前 10 个热点
```

**优化策略**：专注优化热点函数，收益最大。

---

## 7. 如何减少 Python 程序的内存占用？

**答案**：

### 1. 使用生成器

```python
# 列表：占用大量内存
data = [x ** 2 for x in range(1000000)]

# 生成器：按需生成
data = (x ** 2 for x in range(1000000))
```

### 2. 使用 `__slots__`

```python
class Point:
    __slots__ = ['x', 'y']  # 节省约 40% 内存
```

### 3. 使用 array/numpy

```python
import array
arr = array.array('i', [1, 2, 3])  # 比 list 更紧凑
```

### 4. 及时释放引用

```python
del large_object
```

### 5. 使用弱引用

```python
import weakref
weak_ref = weakref.ref(obj)
```

---

## 8. 生产环境的日志应该怎么配置？

**答案**：

```python
import logging
import sys
import os

def setup_production_logging():
    log_level = os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到 stdout（容器日志）
        ],
    )

    # 降低第三方库日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### 最佳实践

1. **级别**：生产环境用 INFO 或 WARNING
2. **格式**：包含时间、模块、级别
3. **输出**：stdout（便于容器收集）
4. **结构化**：使用 JSON 格式便于分析
5. **第三方库**：降低日志级别

---

## 9. cProfile 输出中 cumtime 和 tottime 的区别？

**答案**：

- **tottime**: 函数自身执行时间（不含子函数）
- **cumtime**: 累计执行时间（含子函数）

```
   ncalls  tottime  cumtime  filename:lineno(function)
      100    0.050    0.350  module.py:10(func_a)
            │        │
            │        └── 含子函数总耗时 0.35s
            └── func_a 自身只花了 0.05s
```

**应用**：
- 看 **cumtime** 找整体慢的函数
- 看 **tottime** 找自身慢的函数

---

## 10. breakpoint() 和 pdb.set_trace() 的区别？

**答案**：

| 特性 | breakpoint() | pdb.set_trace() |
|------|-------------|-----------------|
| Python 版本 | 3.7+ | 所有版本 |
| 可配置 | ✓ PYTHONBREAKPOINT | ✗ |
| 可禁用 | ✓ PYTHONBREAKPOINT=0 | ✗ |
| 自定义调试器 | ✓ | ✗ |

```bash
# 禁用所有断点
PYTHONBREAKPOINT=0 python script.py

# 使用 ipdb
PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

**推荐**：使用 breakpoint()（Python 3.7+）。

