# concurrent.futures

> 统一的线程池和进程池接口

## 概述

`concurrent.futures` 提供了高级的异步执行接口，统一了线程池和进程池的使用方式。

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
```

---

## ThreadPoolExecutor

### 基础用法

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(name: str, delay: float) -> str:
    time.sleep(delay)
    return f"Task {name} completed"

# 方式 1：submit 单个任务
with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(task, "A", 1)
    result = future.result()  # 阻塞等待结果
    print(result)

# 方式 2：map 批量任务
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(task, ["A", "B", "C"], [1, 2, 3])
    for result in results:
        print(result)
```

### submit 和 Future

```python
from concurrent.futures import ThreadPoolExecutor, Future

def task(x: int) -> int:
    return x ** 2

with ThreadPoolExecutor(max_workers=4) as executor:
    # submit 返回 Future 对象
    future: Future = executor.submit(task, 5)

    # Future 方法
    print(future.done())       # 是否完成
    print(future.running())    # 是否运行中
    print(future.cancelled())  # 是否被取消

    # 获取结果（阻塞）
    result = future.result()        # 无限等待
    result = future.result(timeout=5)  # 最多等 5 秒

    # 获取异常
    exception = future.exception()  # 如果有异常

    # 取消任务（未开始时可取消）
    future.cancel()
```

### 回调函数

```python
from concurrent.futures import ThreadPoolExecutor

def task(x: int) -> int:
    return x ** 2

def on_complete(future):
    if future.exception():
        print(f"Error: {future.exception()}")
    else:
        print(f"Result: {future.result()}")

with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(task, 5)
    future.add_done_callback(on_complete)
```

---

## ProcessPoolExecutor

用法与 ThreadPoolExecutor 相同，但使用多进程。

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive(n: int) -> int:
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_intensive, 1000000) for _ in range(4)]
        results = [f.result() for f in futures]
        print(results)
```

### 注意事项

```python
# ProcessPoolExecutor 的限制：
# 1. 必须在 if __name__ == "__main__" 中使用
# 2. 任务函数必须是模块级别的（不能是 lambda 或局部函数）
# 3. 参数和返回值必须可序列化（pickle）

# ❌ 错误
executor.submit(lambda x: x ** 2, 5)

# ✅ 正确
def square(x):
    return x ** 2
executor.submit(square, 5)
```

---

## as_completed - 按完成顺序处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def task(n: int) -> int:
    delay = random.random()
    time.sleep(delay)
    return n

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(task, i): i for i in range(10)}

    # 按完成顺序处理
    for future in as_completed(futures):
        task_id = futures[future]
        try:
            result = future.result()
            print(f"Task {task_id} returned {result}")
        except Exception as e:
            print(f"Task {task_id} raised {e}")
```

### 带超时的 as_completed

```python
from concurrent.futures import as_completed, TimeoutError

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, i) for i in range(10)]

    try:
        for future in as_completed(futures, timeout=5):
            print(future.result())
    except TimeoutError:
        print("Some tasks did not complete in time")
```

---

## wait - 等待多个 Future

```python
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED
import time

def task(n: int) -> int:
    time.sleep(n)
    return n

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, i) for i in [3, 1, 2]]

    # 等待第一个完成
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    print(f"First done: {[f.result() for f in done]}")

    # 等待第一个异常
    done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

    # 等待全部完成（默认）
    done, not_done = wait(futures, return_when=ALL_COMPLETED)

    # 带超时
    done, not_done = wait(futures, timeout=2)
```

---

## map - 批量执行

```python
from concurrent.futures import ThreadPoolExecutor

def process(item: int) -> int:
    return item ** 2

with ThreadPoolExecutor(max_workers=4) as executor:
    # map 返回迭代器，按输入顺序返回结果
    results = executor.map(process, range(10))

    # 结果按顺序
    print(list(results))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    # 带超时
    results = executor.map(process, range(10), timeout=5)

    # 多参数
    def add(a, b):
        return a + b

    results = executor.map(add, [1, 2, 3], [10, 20, 30])
    print(list(results))  # [11, 22, 33]
```

### map vs submit

| 特性 | map | submit |
|------|-----|--------|
| 返回 | 迭代器（按输入顺序）| Future |
| 异常处理 | 迭代时抛出 | result() 时抛出 |
| 灵活性 | 较低 | 较高 |
| 适用 | 批量相同操作 | 复杂控制 |

---

## 异常处理

```python
from concurrent.futures import ThreadPoolExecutor

def might_fail(x: int) -> int:
    if x == 2:
        raise ValueError(f"Failed for {x}")
    return x ** 2

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(might_fail, i) for i in range(5)]

    for future in futures:
        try:
            result = future.result()
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")
```

---

## 与 asyncio 配合

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_io(n: int) -> int:
    import time
    time.sleep(1)
    return n * 2

async def main():
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # 在线程池中运行阻塞函数
    result = await loop.run_in_executor(executor, blocking_io, 5)
    print(result)

    # 多个任务
    tasks = [
        loop.run_in_executor(executor, blocking_io, i)
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
```

---

## 选型指南

| 场景 | 推荐 |
|------|------|
| IO 密集（网络、文件）| ThreadPoolExecutor |
| CPU 密集（计算）| ProcessPoolExecutor |
| 异步 + 阻塞混合 | run_in_executor |
| 简单批量处理 | map |
| 复杂控制流 | submit + as_completed |

---

## 实用示例

### 并发下载

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

def download(url: str) -> tuple[str, int]:
    with urllib.request.urlopen(url) as response:
        data = response.read()
        return url, len(data)

urls = [
    "https://www.python.org",
    "https://www.google.com",
    "https://www.github.com",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    future_to_url = {executor.submit(download, url): url for url in urls}

    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            url, size = future.result()
            print(f"{url}: {size} bytes")
        except Exception as e:
            print(f"{url}: failed ({e})")
```

### 带进度的批量处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process(item: int) -> int:
    import time
    time.sleep(0.1)
    return item ** 2

items = list(range(100))

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process, item) for item in items]

    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
```

### 带重试的执行

```python
from concurrent.futures import ThreadPoolExecutor
import time

def with_retry(executor, fn, args, max_retries=3):
    for attempt in range(max_retries):
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=10)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

with ThreadPoolExecutor(max_workers=4) as executor:
    result = with_retry(executor, some_task, (arg1, arg2))
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| executor 不关闭 | 资源泄漏 | 使用 with 语句 |
| map 结果不消费 | 实际未执行 | 遍历或 list() |
| ProcessPool lambda | 无法序列化 | 使用普通函数 |
| 忘记处理异常 | 异常被吞 | try-except |

---

## 小结

1. **ThreadPoolExecutor**: IO 密集型任务
2. **ProcessPoolExecutor**: CPU 密集型任务
3. **submit**: 单个任务，返回 Future
4. **map**: 批量任务，按顺序返回
5. **as_completed**: 按完成顺序处理
6. **wait**: 等待条件控制

