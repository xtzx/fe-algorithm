# 03. GIL 与并发

## 本节目标

- 理解 GIL 是什么以及为什么存在
- 掌握 threading、multiprocessing、asyncio 的选择
- 对比 JavaScript 的并发模型

---

## 什么是 GIL

**GIL (Global Interpreter Lock)** 是 CPython 解释器的全局锁。

```
同一时刻只有一个线程能执行 Python 字节码
```

### 为什么需要 GIL

1. **简化内存管理**: 引用计数不需要加锁
2. **历史原因**: CPython 设计时的权衡
3. **C 扩展安全**: 很多 C 库不是线程安全的

### GIL 的影响

```python
import threading
import time

counter = 0

def increment():
    global counter
    for _ in range(1000000):
        counter += 1

# 单线程
start = time.time()
increment()
increment()
print(f"单线程: {time.time() - start:.2f}s, counter={counter}")

# 多线程
counter = 0
t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)

start = time.time()
t1.start()
t2.start()
t1.join()
t2.join()
print(f"多线程: {time.time() - start:.2f}s, counter={counter}")

# 结果：多线程可能更慢，counter 可能不是 2000000
```

---

## 与 JavaScript 对比

| 特性 | Python | JavaScript |
|------|--------|------------|
| 线程模型 | 多线程 + GIL | 单线程 |
| 并发方式 | threading/multiprocessing/asyncio | 事件循环 + Worker |
| CPU 密集 | multiprocessing | Worker Threads |
| I/O 密集 | asyncio 或 threading | Promise/async-await |

### JavaScript 事件循环 vs Python asyncio

```javascript
// JavaScript
async function fetchData() {
    const response = await fetch(url);
    return response.json();
}
```

```python
# Python
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

---

## threading - 多线程

适合 **I/O 密集型** 任务。

```python
import threading
import time
import requests

def download(url):
    print(f"开始下载: {url}")
    time.sleep(1)  # 模拟网络请求
    print(f"完成: {url}")

urls = [f"https://example.com/{i}" for i in range(5)]

# 多线程下载
threads = []
start = time.time()

for url in urls:
    t = threading.Thread(target=download, args=(url,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"总耗时: {time.time() - start:.2f}s")  # 约 1 秒
```

### 线程池

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=4) as executor:
    # map 方式
    results = list(executor.map(task, range(8)))
    print(results)

    # submit 方式
    future = executor.submit(task, 10)
    print(future.result())  # 等待结果
```

### 线程同步

```python
import threading

# 锁
lock = threading.Lock()
counter = 0

def safe_increment():
    global counter
    with lock:  # 自动获取和释放
        counter += 1

# 信号量
semaphore = threading.Semaphore(3)  # 最多 3 个并发

# 事件
event = threading.Event()
event.set()    # 设置
event.clear()  # 清除
event.wait()   # 等待
```

---

## multiprocessing - 多进程

适合 **CPU 密集型** 任务。

```python
import multiprocessing
import time

def cpu_bound(n):
    """CPU 密集型任务"""
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == "__main__":
    numbers = [10**7] * 4

    # 单进程
    start = time.time()
    results = [cpu_bound(n) for n in numbers]
    print(f"单进程: {time.time() - start:.2f}s")

    # 多进程
    start = time.time()
    with multiprocessing.Pool(4) as pool:
        results = pool.map(cpu_bound, numbers)
    print(f"多进程: {time.time() - start:.2f}s")
```

### 进程池

```python
from concurrent.futures import ProcessPoolExecutor

def compute(x):
    return x ** 2

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute, range(10)))
        print(results)
```

### 进程间通信

```python
from multiprocessing import Process, Queue, Pipe

# Queue
def producer(q):
    q.put("message")

def consumer(q):
    print(q.get())

q = Queue()
p1 = Process(target=producer, args=(q,))
p2 = Process(target=consumer, args=(q,))

# Pipe
parent_conn, child_conn = Pipe()
```

---

## asyncio - 异步 I/O

适合 **高并发 I/O** 任务。

```python
import asyncio

async def fetch(url):
    print(f"开始: {url}")
    await asyncio.sleep(1)  # 模拟网络请求
    print(f"完成: {url}")
    return f"结果: {url}"

async def main():
    urls = [f"https://example.com/{i}" for i in range(5)]

    # 并发执行
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())  # 约 1 秒完成
```

### async 本质

```python
# async 函数返回协程对象
async def hello():
    return "Hello"

coro = hello()
print(type(coro))  # <class 'coroutine'>

# 必须用 await 或事件循环执行
result = asyncio.run(hello())
```

### asyncio 常用模式

```python
import asyncio

async def main():
    # 并发执行多个任务
    results = await asyncio.gather(
        task1(),
        task2(),
        task3(),
    )

    # 超时控制
    try:
        result = await asyncio.wait_for(slow_task(), timeout=5.0)
    except asyncio.TimeoutError:
        print("超时")

    # 创建任务
    task = asyncio.create_task(background_job())

    # 等待第一个完成
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
```

---

## 选择指南

### 决策树

```
需要并发？
├── I/O 密集型
│   ├── 高并发 (100+) → asyncio
│   └── 简单场景 → threading
└── CPU 密集型 → multiprocessing
```

### 对照表

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 网络请求 (少) | threading | 简单 |
| 网络请求 (多) | asyncio | 高效 |
| 文件 I/O | threading | GIL 影响小 |
| 数学计算 | multiprocessing | 绑各自 GIL |
| 图像处理 | multiprocessing | CPU 密集 |
| Web 服务 | asyncio | 高并发 |
| 混合场景 | 进程池 + asyncio | 最佳组合 |

### 混合使用

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_task(n):
    return sum(i*i for i in range(n))

async def main():
    loop = asyncio.get_event_loop()

    # 在进程池中执行 CPU 密集型任务
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_task, 10**7)
        print(result)

asyncio.run(main())
```

---

## GIL 的未来

### Python 3.12+: 子解释器

```python
# 实验性功能
# 每个子解释器有独立的 GIL
```

### Python 3.13+: 无 GIL 模式

```bash
# 编译时可选禁用 GIL（实验性）
./configure --disable-gil
```

### 现在的替代方案

- **PyPy**: 有 GIL 但更快
- **Cython**: 可以释放 GIL
- **NumPy**: C 扩展绑过 GIL
- **多进程**: 完全避开 GIL

---

## 本节要点

1. **GIL**: 同一时刻只有一个线程执行 Python 字节码
2. **threading**: I/O 密集型，GIL 在等待时释放
3. **multiprocessing**: CPU 密集型，绑过 GIL
4. **asyncio**: 高并发 I/O，单线程协作式
5. **选择标准**: I/O 密集用 async/threading，CPU 密集用多进程

