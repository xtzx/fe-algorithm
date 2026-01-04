# threading 基础

> Python 多线程编程入门

## 什么是线程

线程是程序执行的最小单元，同一进程内的线程共享内存空间。

```python
import threading

def task():
    print(f"Running in thread: {threading.current_thread().name}")

# 创建并启动线程
thread = threading.Thread(target=task)
thread.start()
thread.join()  # 等待线程完成
```

---

## 创建线程

### 方式一：函数方式

```python
import threading
import time

def worker(name: str, delay: float) -> None:
    print(f"{name} starting")
    time.sleep(delay)
    print(f"{name} finished")

# 创建线程
t1 = threading.Thread(target=worker, args=("Thread-1", 1))
t2 = threading.Thread(target=worker, args=("Thread-2", 2))

# 启动线程
t1.start()
t2.start()

# 等待完成
t1.join()
t2.join()

print("All threads finished")
```

### 方式二：类继承方式

```python
import threading
import time

class WorkerThread(threading.Thread):
    def __init__(self, name: str, delay: float):
        super().__init__()
        self.delay = delay
        self._name = name

    def run(self) -> None:
        """重写 run 方法"""
        print(f"{self._name} starting")
        time.sleep(self.delay)
        print(f"{self._name} finished")

# 使用
t1 = WorkerThread("Worker-1", 1)
t2 = WorkerThread("Worker-2", 2)
t1.start()
t2.start()
t1.join()
t2.join()
```

---

## 线程参数

```python
import threading

def task(a: int, b: int, name: str = "default") -> None:
    print(f"{name}: {a} + {b} = {a + b}")

# args: 位置参数（元组）
# kwargs: 关键字参数（字典）
thread = threading.Thread(
    target=task,
    args=(1, 2),
    kwargs={"name": "Calculator"}
)
thread.start()
```

---

## 守护线程 daemon

守护线程在主线程结束时自动终止。

```python
import threading
import time

def background_task():
    while True:
        print("Background running...")
        time.sleep(1)

# 普通线程：主线程结束后会等待它
normal = threading.Thread(target=background_task)
# normal.start()  # 程序不会结束

# 守护线程：主线程结束时自动终止
daemon = threading.Thread(target=background_task, daemon=True)
daemon.start()

time.sleep(3)
print("Main thread ending")
# 程序结束，守护线程被终止
```

### 何时使用守护线程

| 场景 | 使用 |
|------|------|
| 后台日志 | ✅ 守护线程 |
| 心跳检测 | ✅ 守护线程 |
| 数据处理 | ❌ 普通线程 |
| 关键任务 | ❌ 普通线程 |

---

## join() - 等待线程

```python
import threading
import time

def task(n: int) -> None:
    time.sleep(n)
    print(f"Task {n} done")

threads = []
for i in range(3):
    t = threading.Thread(target=task, args=(i,))
    t.start()
    threads.append(t)

# 等待所有线程完成
for t in threads:
    t.join()

print("All tasks completed")

# join 可以设置超时
thread = threading.Thread(target=lambda: time.sleep(10))
thread.start()
thread.join(timeout=2)  # 最多等待 2 秒
if thread.is_alive():
    print("Thread still running after timeout")
```

---

## 线程命名和识别

```python
import threading

# 设置线程名
thread = threading.Thread(target=task, name="MyThread")

# 获取当前线程
current = threading.current_thread()
print(current.name)  # 线程名
print(current.ident)  # 线程 ID

# 获取主线程
main = threading.main_thread()

# 获取所有活跃线程
for t in threading.enumerate():
    print(f"{t.name}: alive={t.is_alive()}")

# 活跃线程数
print(threading.active_count())
```

---

## 线程局部变量

`threading.local()` 为每个线程提供独立的变量副本。

```python
import threading

# 创建线程局部存储
local_data = threading.local()

def worker(value: int) -> None:
    # 每个线程有自己的 local_data.value
    local_data.value = value
    print(f"Thread {threading.current_thread().name}: {local_data.value}")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# 主线程没有设置 value
try:
    print(local_data.value)
except AttributeError:
    print("Main thread has no value")
```

### 使用场景

```python
import threading

# 数据库连接池
class ConnectionPool:
    _local = threading.local()

    @classmethod
    def get_connection(cls):
        if not hasattr(cls._local, "conn"):
            cls._local.conn = create_connection()
        return cls._local.conn

# 每个线程都有自己的连接
def worker():
    conn = ConnectionPool.get_connection()
    # 使用连接...
```

---

## GIL 的影响

Python 的 GIL（Global Interpreter Lock）限制了多线程的并行性。

```python
import threading
import time

def cpu_bound(n: int) -> int:
    """CPU 密集型任务"""
    count = 0
    for i in range(n):
        count += i
    return count

def io_bound(seconds: float) -> None:
    """IO 密集型任务"""
    time.sleep(seconds)

# CPU 密集型：多线程不会更快（GIL 限制）
# IO 密集型：多线程有效（等待时释放 GIL）
```

详见 [GIL 深度解析](./14-gil-deep-dive.md)

---

## JS 对照

| Python | JavaScript | 说明 |
|--------|------------|------|
| `threading.Thread` | `Worker` | 线程/Worker |
| `thread.start()` | `worker.postMessage()` | 启动 |
| `thread.join()` | `await worker.terminate()` | 等待 |
| `threading.local()` | 无需（Worker 隔离） | 线程局部 |
| GIL | 无（单线程） | 并发限制 |

```javascript
// JavaScript Worker
const worker = new Worker('worker.js');
worker.postMessage({ data: 'hello' });
worker.onmessage = (e) => console.log(e.data);
```

```python
# Python Thread
import threading

def task(data):
    print(f"Received: {data}")

thread = threading.Thread(target=task, args=("hello",))
thread.start()
thread.join()
```

---

## 实用示例

### 并发下载

```python
import threading
import urllib.request

def download(url: str, filename: str) -> None:
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

urls = [
    ("https://example.com/file1.txt", "file1.txt"),
    ("https://example.com/file2.txt", "file2.txt"),
    ("https://example.com/file3.txt", "file3.txt"),
]

threads = []
for url, filename in urls:
    t = threading.Thread(target=download, args=(url, filename))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("All downloads completed")
```

### 带返回值的线程

```python
import threading
from typing import Any

class ResultThread(threading.Thread):
    """带返回值的线程"""

    def __init__(self, target, args=()):
        super().__init__()
        self._target = target
        self._args = args
        self.result: Any = None
        self.exception: Exception | None = None

    def run(self):
        try:
            self.result = self._target(*self._args)
        except Exception as e:
            self.exception = e

def calculate(n: int) -> int:
    return sum(range(n))

# 使用
thread = ResultThread(target=calculate, args=(1000000,))
thread.start()
thread.join()

if thread.exception:
    raise thread.exception
print(f"Result: {thread.result}")
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 join | 主线程提前结束 | 等待所有线程 |
| 共享变量竞争 | 数据不一致 | 使用锁（见下一章）|
| CPU 密集多线程 | GIL 导致无加速 | 使用多进程 |
| 守护线程写文件 | 数据可能丢失 | 确保刷新/关闭 |

---

## 小结

1. **Thread 创建**：函数方式或继承方式
2. **守护线程**：随主线程结束而终止
3. **join()**：等待线程完成
4. **local()**：线程局部存储
5. **GIL 限制**：CPU 密集用多进程，IO 密集用多线程

