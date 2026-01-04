# GIL 深度解析

> 理解 Python 的全局解释器锁

## 什么是 GIL

GIL（Global Interpreter Lock）是 CPython 解释器中的互斥锁，确保同一时刻只有一个线程执行 Python 字节码。

```
┌──────────────────────────────────────┐
│          Python 进程                  │
│  ┌──────┐ ┌──────┐ ┌──────┐         │
│  │线程1 │ │线程2 │ │线程3 │         │
│  └──┬───┘ └──┬───┘ └──┬───┘         │
│     │        │        │              │
│     └────────┼────────┘              │
│              │                       │
│         ┌────▼────┐                  │
│         │   GIL   │ ← 全局锁         │
│         └────┬────┘                  │
│              │                       │
│         ┌────▼────┐                  │
│         │ CPython │                  │
│         │解释器    │                  │
│         └─────────┘                  │
└──────────────────────────────────────┘
```

---

## GIL 的工作原理

### 字节码执行

```python
# Python 代码
counter += 1

# 编译为字节码（多条指令）
LOAD_GLOBAL   counter
LOAD_CONST    1
BINARY_ADD
STORE_GLOBAL  counter

# GIL 不保证这些指令原子执行
# 线程可能在任意指令间切换
```

### 释放时机

GIL 在以下情况释放：
1. **每执行一定数量的字节码**：默认每 100 条（Python 3.2+ 改为每 5ms）
2. **IO 操作**：文件读写、网络请求
3. **调用 C 扩展**：如果 C 代码显式释放 GIL

```python
# IO 时释放 GIL
with open("file.txt") as f:
    data = f.read()  # 阻塞时释放 GIL

# time.sleep 释放 GIL
import time
time.sleep(1)  # 释放 GIL，其他线程可执行

# numpy 等 C 扩展可能释放 GIL
import numpy as np
result = np.dot(a, b)  # 计算时可能释放 GIL
```

---

## GIL 的影响

### CPU 密集型任务

```python
import threading
import time

def cpu_bound(n):
    count = 0
    for i in range(n):
        count += i
    return count

# 单线程
start = time.time()
cpu_bound(50000000)
cpu_bound(50000000)
print(f"Single thread: {time.time() - start:.2f}s")

# 多线程（不会更快！）
start = time.time()
t1 = threading.Thread(target=cpu_bound, args=(50000000,))
t2 = threading.Thread(target=cpu_bound, args=(50000000,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Two threads: {time.time() - start:.2f}s")

# 结果：多线程可能更慢（线程切换开销）
```

### IO 密集型任务

```python
import threading
import time

def io_bound(delay):
    time.sleep(delay)

# 单线程
start = time.time()
io_bound(1)
io_bound(1)
print(f"Single thread: {time.time() - start:.2f}s")  # ~2s

# 多线程（会更快！）
start = time.time()
t1 = threading.Thread(target=io_bound, args=(1,))
t2 = threading.Thread(target=io_bound, args=(1,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Two threads: {time.time() - start:.2f}s")  # ~1s
```

---

## 并发选型决策树

```
                    任务类型
                       │
         ┌─────────────┴─────────────┐
         │                           │
    CPU 密集型                    IO 密集型
         │                           │
         │                  ┌────────┴────────┐
         │                  │                 │
    multiprocessing      threading         asyncio
    (真正并行)          (并发等待)        (高并发)
```

### 选型指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| CPU 密集计算 | multiprocessing | 绑各自 GIL |
| 文件读写 | threading | IO 时释放 GIL |
| 网络请求（少量）| threading | 简单易用 |
| 网络请求（大量）| asyncio | 更高效 |
| 混合任务 | 进程池 + asyncio | 最佳组合 |

---

## 绕过 GIL 的方法

### 1. 使用 multiprocessing

```python
from multiprocessing import Pool

def cpu_bound(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with Pool(4) as pool:
        results = pool.map(cpu_bound, [10000000] * 4)
```

### 2. 使用 C 扩展

```python
# numpy 等库的 C 实现会释放 GIL
import numpy as np

# 这些操作在 C 层面并行
a = np.random.rand(10000, 10000)
b = np.random.rand(10000, 10000)
c = np.dot(a, b)  # 多核并行
```

### 3. 使用其他解释器

```python
# PyPy：JIT 编译，某些场景更快
# Jython：基于 JVM，无 GIL
# IronPython：基于 .NET，无 GIL

# 但这些有各自的限制：
# - 不完全兼容 CPython
# - 某些 C 扩展不可用
```

### 4. 使用 Cython 释放 GIL

```cython
# mymodule.pyx
cimport cython

@cython.boundscheck(False)
def cpu_intensive():
    cdef int i, total = 0
    with nogil:  # 释放 GIL
        for i in range(100000000):
            total += i
    return total
```

---

## Python 3.13+ free-threading

### PEP 703 概述

Python 3.13 开始实验性支持禁用 GIL。

```bash
# 编译时启用
./configure --disable-gil

# 检查是否支持
python -c "import sys; print(sys._is_gil_enabled())"
```

### 使用方式

```python
# Python 3.13+ 可能的 API
import sys

# 检查 GIL 状态
if hasattr(sys, '_is_gil_enabled'):
    print(f"GIL enabled: {sys._is_gil_enabled()}")
```

### 当前状态

- Python 3.13：实验性支持（需要特殊编译）
- 向后兼容：默认仍启用 GIL
- 性能影响：单线程可能略慢
- 生态系统：C 扩展需要适配

---

## GIL 常见误解

### 误解 1：GIL 使 Python 单线程

```python
# 错误！Python 支持多线程
# GIL 只是限制了并行执行字节码
# IO 等待时可以切换线程

import threading
threads = [threading.Thread(target=io_task) for _ in range(10)]
# 这些线程可以并发执行 IO
```

### 误解 2：GIL 确保线程安全

```python
# 错误！GIL 不保证原子操作
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # 仍需要锁！

# counter += 1 是多条字节码，线程可能在中间切换
```

### 误解 3：多线程总是没用

```python
# 错误！IO 密集型任务多线程有效
import threading
import requests

urls = ["https://..."] * 100

# 单线程：串行请求，很慢
# 多线程：并发请求，快得多
```

---

## 性能对比

```python
import time
import threading
from multiprocessing import Pool

def cpu_task(n):
    return sum(i * i for i in range(n))

N = 10000000

# 基准：单线程
start = time.time()
for _ in range(4):
    cpu_task(N)
single = time.time() - start

# 多线程
start = time.time()
threads = [threading.Thread(target=cpu_task, args=(N,)) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
multi_thread = time.time() - start

# 多进程
if __name__ == "__main__":
    start = time.time()
    with Pool(4) as pool:
        pool.map(cpu_task, [N] * 4)
    multi_process = time.time() - start

    print(f"Single thread:  {single:.2f}s")
    print(f"4 threads:      {multi_thread:.2f}s")  # 差不多或更慢
    print(f"4 processes:    {multi_process:.2f}s")  # 快 ~4x
```

---

## 最佳实践

### 1. 首先判断任务类型

```python
def choose_strategy(task_type):
    if task_type == "cpu_bound":
        return "multiprocessing"
    elif task_type == "io_bound_few":
        return "threading"
    elif task_type == "io_bound_many":
        return "asyncio"
    else:
        return "single_thread"
```

### 2. 混合策略

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def main():
    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=4)

    # CPU 密集任务放进程池
    cpu_result = await loop.run_in_executor(executor, cpu_task, arg)

    # IO 密集任务用协程
    io_result = await async_io_task()

asyncio.run(main())
```

### 3. 利用 C 扩展

```python
import numpy as np

# 大量数值计算用 numpy
# 底层 C 代码会释放 GIL 并行执行
result = np.dot(large_matrix_a, large_matrix_b)
```

---

## 小结

| 要点 | 说明 |
|------|------|
| GIL 是什么 | CPython 的全局锁 |
| 影响 | 同时只有一个线程执行字节码 |
| IO 密集 | 多线程有效（IO 时释放 GIL）|
| CPU 密集 | 多线程无效，用多进程 |
| 绕过方法 | multiprocessing、C 扩展、其他解释器 |
| 未来 | Python 3.13+ free-threading |

