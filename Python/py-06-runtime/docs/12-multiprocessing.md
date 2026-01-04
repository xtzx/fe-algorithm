# multiprocessing - 多进程

> 绕过 GIL，真正的并行计算

## 为什么需要多进程

Python 的 GIL 限制了多线程在 CPU 密集型任务上的并行性。多进程可以绑各自的 GIL，实现真正的并行。

```python
# 多线程：受 GIL 限制，无法并行
# 多进程：每个进程有自己的 GIL，可以并行
```

---

## Process 基础

### 创建进程

```python
from multiprocessing import Process
import os

def worker(name: str):
    print(f"Worker {name}, PID: {os.getpid()}")

if __name__ == "__main__":  # Windows 必须！
    processes = []
    for i in range(4):
        p = Process(target=worker, args=(f"P{i}",))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### 类继承方式

```python
from multiprocessing import Process

class MyProcess(Process):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self):
        print(f"Process {self.name} running")

if __name__ == "__main__":
    p = MyProcess("Worker")
    p.start()
    p.join()
```

---

## 进程池 Pool

管理多个工作进程，自动分配任务。

### map - 批量处理

```python
from multiprocessing import Pool

def square(x: int) -> int:
    return x ** 2

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        results = pool.map(square, range(10))
        print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### apply_async - 异步提交

```python
from multiprocessing import Pool

def task(x: int) -> int:
    return x * 2

if __name__ == "__main__":
    with Pool(4) as pool:
        # 异步提交
        result = pool.apply_async(task, (10,))

        # 做其他事情...

        # 获取结果
        print(result.get())  # 20

        # 带超时
        print(result.get(timeout=5))
```

### 批量异步提交

```python
from multiprocessing import Pool

def process_item(item):
    return item ** 2

if __name__ == "__main__":
    items = range(100)

    with Pool(4) as pool:
        # 方式 1：map（阻塞，返回列表）
        results = pool.map(process_item, items)

        # 方式 2：imap（惰性，返回迭代器）
        for result in pool.imap(process_item, items):
            print(result)

        # 方式 3：imap_unordered（无序，更快）
        for result in pool.imap_unordered(process_item, items):
            print(result)

        # 方式 4：starmap（多参数）
        pairs = [(1, 2), (3, 4), (5, 6)]
        results = pool.starmap(lambda a, b: a + b, pairs)
```

### 回调函数

```python
from multiprocessing import Pool

def task(x):
    return x * 2

def callback(result):
    print(f"Got result: {result}")

def error_callback(error):
    print(f"Got error: {error}")

if __name__ == "__main__":
    with Pool(4) as pool:
        pool.apply_async(
            task,
            (10,),
            callback=callback,
            error_callback=error_callback
        )
        pool.close()
        pool.join()
```

---

## 进程间通信

进程不共享内存，需要特殊机制通信。

### Queue - 队列

```python
from multiprocessing import Process, Queue

def producer(q: Queue):
    for i in range(5):
        q.put(i)
        print(f"Produced: {i}")
    q.put(None)  # 结束信号

def consumer(q: Queue):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")

if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

### Pipe - 管道

```python
from multiprocessing import Process, Pipe

def sender(conn):
    conn.send("Hello from sender")
    conn.send([1, 2, 3])
    conn.close()

def receiver(conn):
    print(conn.recv())  # "Hello from sender"
    print(conn.recv())  # [1, 2, 3]

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()

    p1 = Process(target=sender, args=(child_conn,))
    p2 = Process(target=receiver, args=(parent_conn,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

### Queue vs Pipe

| 特性 | Queue | Pipe |
|------|-------|------|
| 通信方式 | 多生产者-多消费者 | 双端通信 |
| 性能 | 较慢 | 较快 |
| 复杂度 | 简单 | 需要管理两端 |

---

## 共享内存

### Value - 单个值

```python
from multiprocessing import Process, Value

def increment(counter):
    for _ in range(10000):
        with counter.get_lock():  # 需要锁！
            counter.value += 1

if __name__ == "__main__":
    counter = Value('i', 0)  # 'i' = int

    processes = [Process(target=increment, args=(counter,)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"Result: {counter.value}")  # 40000
```

### Array - 数组

```python
from multiprocessing import Process, Array

def fill_array(arr, start):
    for i in range(len(arr)):
        arr[i] = start + i

if __name__ == "__main__":
    arr = Array('i', 10)  # 10 个 int

    p = Process(target=fill_array, args=(arr, 100))
    p.start()
    p.join()

    print(list(arr))  # [100, 101, 102, ...]
```

### 类型代码

| 代码 | C 类型 | Python 类型 |
|------|--------|-------------|
| 'b' | signed char | int |
| 'B' | unsigned char | int |
| 'i' | signed int | int |
| 'I' | unsigned int | int |
| 'f' | float | float |
| 'd' | double | float |

### shared_memory（Python 3.8+）

```python
from multiprocessing import shared_memory
import numpy as np

# 创建共享内存
shm = shared_memory.SharedMemory(create=True, size=1000)

# 使用 numpy 操作
arr = np.ndarray((10, 10), dtype=np.float64, buffer=shm.buf)
arr[:] = 0  # 初始化

# 在另一个进程中访问
def worker(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray((10, 10), dtype=np.float64, buffer=existing_shm.buf)
    arr[0, 0] = 42
    existing_shm.close()

# 清理
shm.close()
shm.unlink()  # 删除共享内存
```

---

## Manager - 共享复杂对象

```python
from multiprocessing import Process, Manager

def worker(shared_dict, shared_list):
    shared_dict["key"] = "value"
    shared_list.append(42)

if __name__ == "__main__":
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()

        p = Process(target=worker, args=(shared_dict, shared_list))
        p.start()
        p.join()

        print(dict(shared_dict))  # {'key': 'value'}
        print(list(shared_list))  # [42]
```

### Manager 支持的类型

- `dict()`
- `list()`
- `Value()`
- `Array()`
- `Queue()`
- `Lock()`
- `Semaphore()`
- `Event()`
- `Condition()`

---

## 进程同步

```python
from multiprocessing import Process, Lock

lock = Lock()

def safe_print(msg):
    with lock:
        print(msg)

if __name__ == "__main__":
    processes = [
        Process(target=safe_print, args=(f"Message {i}",))
        for i in range(5)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
```

---

## 注意事项

### Windows 下的 if __name__ == "__main__"

```python
# ❌ Windows 下会出错
from multiprocessing import Process

def worker():
    print("Working")

p = Process(target=worker)
p.start()  # Windows 下会递归创建进程

# ✅ 正确做法
if __name__ == "__main__":
    p = Process(target=worker)
    p.start()
```

### 序列化限制

```python
# 传递给子进程的数据必须可序列化（pickle）
# ❌ 不能传递：lambda、局部函数、数据库连接等

# ❌ 错误
p = Process(target=lambda: print("hello"))

# ✅ 正确
def worker():
    print("hello")
p = Process(target=worker)
```

---

## 与 GIL 的关系

| 场景 | 多线程 | 多进程 |
|------|--------|--------|
| CPU 密集 | ❌ GIL 限制 | ✅ 真正并行 |
| IO 密集 | ✅ 可用 | ✅ 可用（开销大）|
| 内存共享 | ✅ 自然共享 | ❌ 需要特殊机制 |
| 进程开销 | 低 | 高 |

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 `if __name__` | Windows 递归创建 | 始终使用 |
| 共享变量不同步 | 数据竞争 | 使用锁或 Manager |
| 传递不可序列化对象 | pickle 错误 | 使用普通函数 |
| 忘记 join | 主进程提前结束 | 等待子进程 |
| Pool 不关闭 | 资源泄漏 | 使用 with 或 close/join |

---

## 小结

1. **Process**: 单个进程
2. **Pool**: 进程池，自动分配任务
3. **Queue/Pipe**: 进程间通信
4. **Value/Array**: 共享简单数据
5. **Manager**: 共享复杂对象
6. **Lock**: 进程同步

