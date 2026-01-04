# 线程安全队列

> queue 模块：生产者-消费者模式

## queue 模块

Python 标准库提供线程安全的队列实现。

```python
from queue import Queue, LifoQueue, PriorityQueue

# 先进先出队列
fifo = Queue()

# 后进先出队列（栈）
lifo = LifoQueue()

# 优先级队列
pq = PriorityQueue()
```

---

## Queue 基础

### 基本操作

```python
from queue import Queue

q = Queue()

# 放入元素
q.put("item1")
q.put("item2")

# 取出元素
item = q.get()  # "item1"

# 非阻塞操作
q.put_nowait("item3")       # 等同于 put(block=False)
item = q.get_nowait()       # 等同于 get(block=False)

# 查询
q.qsize()   # 大约的大小（非精确）
q.empty()   # 是否为空
q.full()    # 是否已满
```

### 阻塞和超时

```python
from queue import Queue, Empty, Full

q = Queue(maxsize=2)  # 有界队列

# put 阻塞直到有空间
q.put("item", block=True)   # 默认阻塞
q.put("item", timeout=5)     # 最多等 5 秒

# get 阻塞直到有元素
item = q.get(block=True)    # 默认阻塞
item = q.get(timeout=5)      # 最多等 5 秒

# 异常处理
try:
    q.put("item", block=False)
except Full:
    print("Queue is full")

try:
    item = q.get(block=False)
except Empty:
    print("Queue is empty")
```

---

## 生产者-消费者模式

### 基础实现

```python
import threading
import time
from queue import Queue

def producer(q: Queue, items: list):
    for item in items:
        print(f"Producing: {item}")
        q.put(item)
        time.sleep(0.1)
    print("Producer done")

def consumer(q: Queue, name: str):
    while True:
        item = q.get()
        if item is None:  # 结束信号
            q.task_done()
            break
        print(f"{name} consuming: {item}")
        time.sleep(0.2)
        q.task_done()
    print(f"{name} done")

# 创建队列
q = Queue()

# 启动消费者
consumers = [
    threading.Thread(target=consumer, args=(q, f"Consumer-{i}"))
    for i in range(2)
]
for c in consumers:
    c.start()

# 启动生产者
producer_thread = threading.Thread(
    target=producer,
    args=(q, list(range(10)))
)
producer_thread.start()

# 等待生产者完成
producer_thread.join()

# 发送结束信号
for _ in consumers:
    q.put(None)

# 等待消费者完成
for c in consumers:
    c.join()

print("All done")
```

### task_done 和 join

```python
from queue import Queue
import threading

q = Queue()

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        process(item)
        q.task_done()  # 标记任务完成

# 启动工作线程
threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()

# 添加任务
for item in range(20):
    q.put(item)

# 等待所有任务完成
q.join()  # 阻塞直到所有 task_done() 被调用

# 停止工作线程
for _ in threads:
    q.put(None)
for t in threads:
    t.join()
```

---

## LifoQueue - 后进先出

```python
from queue import LifoQueue

stack = LifoQueue()

stack.put(1)
stack.put(2)
stack.put(3)

print(stack.get())  # 3
print(stack.get())  # 2
print(stack.get())  # 1
```

### 使用场景

- 深度优先搜索
- 撤销操作
- 函数调用栈模拟

---

## PriorityQueue - 优先级队列

按优先级（最小值优先）取出元素。

```python
from queue import PriorityQueue

pq = PriorityQueue()

# 放入 (优先级, 数据) 元组
pq.put((3, "low priority"))
pq.put((1, "high priority"))
pq.put((2, "medium priority"))

# 按优先级取出
print(pq.get())  # (1, "high priority")
print(pq.get())  # (2, "medium priority")
print(pq.get())  # (3, "low priority")
```

### 自定义排序

```python
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class Task:
    priority: int
    data: Any = field(compare=False)  # 不参与比较

pq = PriorityQueue()
pq.put(Task(priority=2, data="Task B"))
pq.put(Task(priority=1, data="Task A"))
pq.put(Task(priority=3, data="Task C"))

while not pq.empty():
    task = pq.get()
    print(f"Processing: {task.data}")
```

### 使用场景

- 任务调度
- 事件处理
- Dijkstra 算法
- 最小堆应用

---

## SimpleQueue（Python 3.7+）

简化版队列，无大小限制，无 task_done/join。

```python
from queue import SimpleQueue

q = SimpleQueue()
q.put("item")
item = q.get()

# 更简单，性能略好
# 但没有 maxsize, task_done(), join()
```

---

## 与 asyncio.Queue 对比

| 特性 | queue.Queue | asyncio.Queue |
|------|-------------|---------------|
| 线程安全 | ✅ | ❌（协程安全）|
| 阻塞方式 | 线程阻塞 | await |
| 使用场景 | 多线程 | 异步编程 |

```python
# 多线程
from queue import Queue
q = Queue()
q.put(item)
item = q.get()  # 阻塞线程

# 异步
import asyncio
q = asyncio.Queue()
await q.put(item)
item = await q.get()  # 挂起协程
```

---

## 实用示例

### 工作池

```python
import threading
from queue import Queue
from typing import Callable, Any

class WorkerPool:
    def __init__(self, num_workers: int):
        self.queue = Queue()
        self.workers = []

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            func, args, kwargs, result_holder = self.queue.get()
            try:
                result = func(*args, **kwargs)
                if result_holder is not None:
                    result_holder.append(result)
            except Exception as e:
                if result_holder is not None:
                    result_holder.append(e)
            finally:
                self.queue.task_done()

    def submit(self, func: Callable, *args, **kwargs):
        self.queue.put((func, args, kwargs, None))

    def wait(self):
        self.queue.join()

# 使用
pool = WorkerPool(4)
for i in range(10):
    pool.submit(print, f"Task {i}")
pool.wait()
```

### 日志队列

```python
import threading
from queue import Queue
import logging

class QueueHandler(logging.Handler):
    """将日志放入队列的 Handler"""

    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        self.queue.put(record)

class QueueListener:
    """从队列读取日志并处理"""

    def __init__(self, queue: Queue, *handlers):
        self.queue = queue
        self.handlers = handlers
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def _monitor(self):
        while not self._stop.is_set():
            try:
                record = self.queue.get(timeout=0.1)
                for handler in self.handlers:
                    handler.handle(record)
            except:
                pass

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 task_done | join() 永久阻塞 | 每次 get 后调用 |
| 无限等待 | 没有超时 | 使用 timeout 参数 |
| qsize 不准确 | 多线程下会变化 | 不要依赖精确值 |
| 无结束信号 | 消费者不知道何时停止 | 使用 None 或特殊值 |

---

## 小结

| 队列类型 | 顺序 | 用途 |
|---------|------|------|
| Queue | 先进先出 | 通用任务队列 |
| LifoQueue | 后进先出 | 栈、DFS |
| PriorityQueue | 优先级 | 调度、排序 |
| SimpleQueue | 先进先出 | 简单场景 |

核心方法：
- `put()` / `get()` - 放入/取出
- `task_done()` / `join()` - 任务完成/等待
- `empty()` / `full()` / `qsize()` - 状态查询

