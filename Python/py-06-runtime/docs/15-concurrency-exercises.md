# 并发编程练习题

## 基础练习

### 练习 1：创建并启动线程

创建 5 个线程，每个打印自己的编号和线程名。

```python
import threading

def worker(thread_id: int) -> None:
    # 你的代码
    pass

# 创建并启动 5 个线程
# 等待所有线程完成
```

<details>
<summary>参考答案</summary>

```python
import threading

def worker(thread_id: int) -> None:
    print(f"Thread {thread_id}: {threading.current_thread().name}")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```
</details>

---

### 练习 2：使用 Lock 保护共享数据

修复以下代码的竞争条件：

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 1000000, Got: {counter}")
```

<details>
<summary>参考答案</summary>

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 1000000, Got: {counter}")  # 现在正确
```
</details>

---

### 练习 3：使用线程池

使用 ThreadPoolExecutor 并发计算 1-100 每个数的平方。

```python
from concurrent.futures import ThreadPoolExecutor

def square(n: int) -> int:
    return n ** 2

# 使用线程池计算并打印结果
```

<details>
<summary>参考答案</summary>

```python
from concurrent.futures import ThreadPoolExecutor

def square(n: int) -> int:
    return n ** 2

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(square, range(1, 101))
    print(list(results))
```
</details>

---

### 练习 4：生产者-消费者

实现一个简单的生产者-消费者模式：
- 1 个生产者产生 20 个数字
- 2 个消费者处理数字

```python
import threading
from queue import Queue

# 实现生产者和消费者
```

<details>
<summary>参考答案</summary>

```python
import threading
from queue import Queue
import time

def producer(q: Queue):
    for i in range(20):
        q.put(i)
        print(f"Produced: {i}")
        time.sleep(0.1)
    # 发送结束信号
    q.put(None)
    q.put(None)

def consumer(q: Queue, name: str):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"{name} consumed: {item}")
        q.task_done()

q = Queue()

prod = threading.Thread(target=producer, args=(q,))
cons1 = threading.Thread(target=consumer, args=(q, "Consumer-1"))
cons2 = threading.Thread(target=consumer, args=(q, "Consumer-2"))

prod.start()
cons1.start()
cons2.start()

prod.join()
cons1.join()
cons2.join()
```
</details>

---

### 练习 5：Event 同步

使用 Event 实现：3 个线程等待信号，主线程 2 秒后发送信号。

```python
import threading

# 实现
```

<details>
<summary>参考答案</summary>

```python
import threading
import time

event = threading.Event()

def waiter(name: str):
    print(f"{name} waiting...")
    event.wait()
    print(f"{name} got signal!")

threads = [
    threading.Thread(target=waiter, args=(f"Thread-{i}",))
    for i in range(3)
]

for t in threads:
    t.start()

print("Main: waiting 2 seconds...")
time.sleep(2)
print("Main: setting event")
event.set()

for t in threads:
    t.join()
```
</details>

---

## 进阶练习

### 练习 6：使用 Semaphore 限制并发

限制同时最多 3 个线程执行任务：

```python
import threading
import time

def task(name: str):
    print(f"{name} starting")
    time.sleep(1)
    print(f"{name} done")

# 创建 10 个线程，但同时最多 3 个执行
```

<details>
<summary>参考答案</summary>

```python
import threading
import time

semaphore = threading.Semaphore(3)

def task(name: str):
    with semaphore:
        print(f"{name} starting")
        time.sleep(1)
        print(f"{name} done")

threads = [
    threading.Thread(target=task, args=(f"Task-{i}",))
    for i in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```
</details>

---

### 练习 7：multiprocessing 计算

使用多进程计算 1 到 1000000 的平方和，分成 4 个进程。

```python
from multiprocessing import Pool

# 实现
```

<details>
<summary>参考答案</summary>

```python
from multiprocessing import Pool

def partial_sum(args):
    start, end = args
    return sum(i * i for i in range(start, end))

if __name__ == "__main__":
    n = 1000000
    chunk_size = n // 4
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size)
        for i in range(4)
    ]

    with Pool(4) as pool:
        results = pool.map(partial_sum, ranges)

    total = sum(results)
    print(f"Sum of squares: {total}")
```
</details>

---

### 练习 8：as_completed 处理

使用 as_completed 按完成顺序处理结果：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

def task(n: int) -> tuple[int, float]:
    delay = random.random()
    time.sleep(delay)
    return n, delay

# 提交 10 个任务，按完成顺序打印
```

<details>
<summary>参考答案</summary>

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

def task(n: int) -> tuple[int, float]:
    delay = random.random()
    time.sleep(delay)
    return n, delay

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(task, i): i for i in range(10)}

    for future in as_completed(futures):
        task_id = futures[future]
        n, delay = future.result()
        print(f"Task {task_id} completed with delay {delay:.2f}s")
```
</details>

---

### 练习 9：线程安全的计数器类

实现一个线程安全的计数器类：

```python
import threading

class ThreadSafeCounter:
    def __init__(self):
        pass

    def increment(self) -> int:
        """增加并返回新值"""
        pass

    def decrement(self) -> int:
        """减少并返回新值"""
        pass

    def value(self) -> int:
        """返回当前值"""
        pass
```

<details>
<summary>参考答案</summary>

```python
import threading

class ThreadSafeCounter:
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

# 测试
counter = ThreadSafeCounter()

def worker():
    for _ in range(10000):
        counter.increment()

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Final value: {counter.value}")  # 100000
```
</details>

---

### 练习 10：带超时的任务执行

实现带超时的任务执行函数：

```python
from concurrent.futures import ThreadPoolExecutor

def run_with_timeout(func, args, timeout: float):
    """
    执行函数，如果超时返回 None
    """
    pass
```

<details>
<summary>参考答案</summary>

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def run_with_timeout(func, args, timeout: float):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None

# 测试
import time

def slow_task(n: int) -> int:
    time.sleep(n)
    return n * 2

result = run_with_timeout(slow_task, (1,), timeout=2)
print(f"Result: {result}")  # 2

result = run_with_timeout(slow_task, (5,), timeout=2)
print(f"Result: {result}")  # None
```
</details>

---

## 挑战练习

### 挑战 1：实现简单的线程池

```python
import threading
from queue import Queue
from typing import Callable

class SimpleThreadPool:
    def __init__(self, num_workers: int):
        """初始化线程池"""
        pass

    def submit(self, func: Callable, *args, **kwargs):
        """提交任务"""
        pass

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        pass
```

<details>
<summary>参考答案</summary>

```python
import threading
from queue import Queue
from typing import Callable, Any

class SimpleThreadPool:
    def __init__(self, num_workers: int):
        self._queue = Queue()
        self._workers = []
        self._shutdown = False

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while True:
            task = self._queue.get()
            if task is None:
                break
            func, args, kwargs = task
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Task error: {e}")
            finally:
                self._queue.task_done()

    def submit(self, func: Callable, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("Pool is shut down")
        self._queue.put((func, args, kwargs))

    def shutdown(self, wait: bool = True):
        self._shutdown = True
        for _ in self._workers:
            self._queue.put(None)
        if wait:
            for t in self._workers:
                t.join()

# 测试
def task(n: int):
    print(f"Processing {n}")

pool = SimpleThreadPool(4)
for i in range(10):
    pool.submit(task, i)
pool.shutdown()
```
</details>

---

### 挑战 2：实现读写锁

```python
import threading

class ReadWriteLock:
    """
    允许多个读者同时读
    写者独占访问
    """

    def __init__(self):
        pass

    def acquire_read(self):
        pass

    def release_read(self):
        pass

    def acquire_write(self):
        pass

    def release_write(self):
        pass
```

<details>
<summary>参考答案</summary>

```python
import threading

class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()

# 使用上下文管理器
class ReadLock:
    def __init__(self, rwlock):
        self._rwlock = rwlock
    def __enter__(self):
        self._rwlock.acquire_read()
    def __exit__(self, *args):
        self._rwlock.release_read()

class WriteLock:
    def __init__(self, rwlock):
        self._rwlock = rwlock
    def __enter__(self):
        self._rwlock.acquire_write()
    def __exit__(self, *args):
        self._rwlock.release_write()
```
</details>

---

### 挑战 3：并发下载器

实现一个并发下载器：
- 限制最大并发数
- 支持重试
- 显示进度

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

def download_files(urls: List[str], max_workers: int = 4, max_retries: int = 3) -> List[Tuple[str, bool]]:
    """
    并发下载文件
    返回：[(url, success), ...]
    """
    pass
```

<details>
<summary>参考答案</summary>

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import time
from typing import List, Tuple

def download_with_retry(url: str, max_retries: int) -> Tuple[str, bool]:
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                _ = response.read()
            return url, True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    return url, False

def download_files(
    urls: List[str],
    max_workers: int = 4,
    max_retries: int = 3
) -> List[Tuple[str, bool]]:
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_with_retry, url, max_retries): url
            for url in urls
        }

        completed = 0
        total = len(urls)

        for future in as_completed(futures):
            url, success = future.result()
            results.append((url, success))
            completed += 1
            status = "✓" if success else "✗"
            print(f"[{completed}/{total}] {status} {url}")

    return results

# 测试
urls = [
    "https://www.python.org",
    "https://www.github.com",
    "https://invalid.url.that.does.not.exist",
]
results = download_files(urls)
```
</details>

---

## 面试题

### 1. GIL 是什么？它如何影响多线程？

<details>
<summary>答案</summary>

GIL（Global Interpreter Lock）是 CPython 中的全局锁，确保同一时刻只有一个线程执行 Python 字节码。

影响：
- CPU 密集型任务：多线程无法并行，性能不会提升
- IO 密集型任务：等待 IO 时释放 GIL，多线程有效

解决方案：
- CPU 密集：使用 multiprocessing
- IO 密集：使用 threading 或 asyncio
</details>

### 2. threading.Lock 和 threading.RLock 的区别？

<details>
<summary>答案</summary>

- `Lock`：普通互斥锁，同一线程重复获取会死锁
- `RLock`：可重入锁，同一线程可以多次获取

使用场景：
- `Lock`：简单互斥
- `RLock`：递归调用或嵌套锁定
</details>

### 3. 如何避免死锁？

<details>
<summary>答案</summary>

1. **固定锁顺序**：所有线程按相同顺序获取锁
2. **使用超时**：`lock.acquire(timeout=1)`
3. **使用 RLock**：避免自我死锁
4. **最小化锁范围**：减少持有锁的时间
5. **使用上下文管理器**：确保释放锁
</details>

### 4. Process 和 Thread 的区别？

<details>
<summary>答案</summary>

| 特性 | Process | Thread |
|------|---------|--------|
| 内存 | 独立 | 共享 |
| 通信 | Queue/Pipe/共享内存 | 直接共享 |
| 开销 | 大 | 小 |
| GIL | 各自独立 | 共享 |
| 安全性 | 高（隔离）| 低（需同步）|
</details>

### 5. concurrent.futures 的优势是什么？

<details>
<summary>答案</summary>

1. **统一接口**：ThreadPoolExecutor 和 ProcessPoolExecutor 使用相同 API
2. **高级抽象**：Future、as_completed、wait 等
3. **上下文管理**：with 语句自动管理
4. **回调支持**：add_done_callback
5. **异常处理**：future.result() 会抛出异常
</details>

