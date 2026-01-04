# 线程同步

> Lock、Condition、Event、Semaphore 等同步原语

## 为什么需要同步

多线程共享数据时，没有同步会导致竞争条件。

```python
import threading

# 危险！竞争条件
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # 非原子操作

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 1000000, Got: {counter}")  # 通常小于预期
```

---

## Lock - 互斥锁

最基本的同步原语，确保同一时间只有一个线程访问资源。

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        lock.acquire()  # 获取锁
        try:
            counter += 1
        finally:
            lock.release()  # 释放锁

# 推荐：使用 with 语句
def increment_safe():
    global counter
    for _ in range(100000):
        with lock:  # 自动获取和释放
            counter += 1

threads = [threading.Thread(target=increment_safe) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Result: {counter}")  # 正确：1000000
```

### Lock 方法

```python
lock = threading.Lock()

# 阻塞获取
lock.acquire()

# 非阻塞获取
if lock.acquire(blocking=False):
    try:
        # 获取成功
        pass
    finally:
        lock.release()
else:
    # 获取失败
    pass

# 带超时获取
if lock.acquire(timeout=1.0):
    try:
        pass
    finally:
        lock.release()

# 检查是否被锁定
print(lock.locked())
```

---

## RLock - 可重入锁

同一线程可以多次获取的锁。

```python
import threading

lock = threading.Lock()
rlock = threading.RLock()

# Lock：同一线程重复获取会死锁
# lock.acquire()
# lock.acquire()  # 死锁！

# RLock：同一线程可以多次获取
rlock.acquire()
rlock.acquire()  # OK
rlock.release()
rlock.release()

# 使用场景：递归函数
class SafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.RLock()

    def increment(self):
        with self._lock:
            self._value += 1

    def increment_twice(self):
        with self._lock:
            self.increment()  # 再次获取同一个锁
            self.increment()
```

---

## Condition - 条件变量

允许线程等待某个条件成立。

```python
import threading
import time

condition = threading.Condition()
data = []

def producer():
    for i in range(5):
        time.sleep(0.5)
        with condition:
            data.append(i)
            print(f"Produced: {i}")
            condition.notify()  # 通知一个等待的线程

def consumer():
    while True:
        with condition:
            while not data:  # 用 while 防止虚假唤醒
                condition.wait()  # 等待通知
            item = data.pop(0)
            print(f"Consumed: {item}")
        if item == 4:
            break

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t2.start()
t1.start()
t1.join()
t2.join()
```

### Condition 方法

```python
condition = threading.Condition()

with condition:
    condition.wait()           # 等待通知
    condition.wait(timeout=1)  # 带超时等待
    condition.notify()         # 通知一个等待线程
    condition.notify_all()     # 通知所有等待线程
```

---

## Event - 事件

简单的线程间信号机制。

```python
import threading
import time

event = threading.Event()

def waiter(name: str):
    print(f"{name} waiting for event...")
    event.wait()  # 阻塞直到事件被设置
    print(f"{name} got the event!")

def setter():
    time.sleep(2)
    print("Setting event")
    event.set()  # 设置事件

threads = [
    threading.Thread(target=waiter, args=(f"Waiter-{i}",))
    for i in range(3)
]
for t in threads:
    t.start()

setter_thread = threading.Thread(target=setter)
setter_thread.start()

for t in threads:
    t.join()
setter_thread.join()
```

### Event 方法

```python
event = threading.Event()

event.set()        # 设置事件（唤醒所有等待者）
event.clear()      # 清除事件
event.is_set()     # 检查是否设置
event.wait()       # 等待事件
event.wait(timeout=1)  # 带超时等待，返回是否成功
```

### Event vs Condition

| 特性 | Event | Condition |
|------|-------|-----------|
| 复杂度 | 简单 | 复杂 |
| 状态 | 二元（set/clear） | 任意条件 |
| 唤醒 | 所有等待者 | 可选一个或所有 |
| 用途 | 一次性信号 | 复杂同步 |

---

## Semaphore - 信号量

限制同时访问资源的线程数。

```python
import threading
import time

# 最多允许 3 个线程同时运行
semaphore = threading.Semaphore(3)

def worker(name: str):
    with semaphore:
        print(f"{name} acquired")
        time.sleep(1)
        print(f"{name} released")

threads = [
    threading.Thread(target=worker, args=(f"Worker-{i}",))
    for i in range(10)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### BoundedSemaphore

防止释放次数超过获取次数。

```python
# 普通 Semaphore
sem = threading.Semaphore(2)
sem.release()  # 计数变为 3，可能不是预期

# BoundedSemaphore
bsem = threading.BoundedSemaphore(2)
# bsem.release()  # ValueError: Semaphore released too many times
```

### 使用场景

```python
import threading

# 连接池
class ConnectionPool:
    def __init__(self, max_connections: int):
        self._semaphore = threading.Semaphore(max_connections)
        self._connections = []

    def get_connection(self):
        self._semaphore.acquire()
        # 返回或创建连接

    def release_connection(self, conn):
        self._semaphore.release()
```

---

## Barrier - 屏障

让多个线程在某个点同步。

```python
import threading
import time
import random

# 3 个线程到达屏障后才继续
barrier = threading.Barrier(3)

def worker(name: str):
    print(f"{name} doing phase 1")
    time.sleep(random.random())

    print(f"{name} waiting at barrier")
    barrier.wait()  # 等待所有线程到达

    print(f"{name} doing phase 2")

threads = [
    threading.Thread(target=worker, args=(f"Worker-{i}",))
    for i in range(3)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Barrier 带动作

```python
def on_barrier():
    print("All threads reached barrier!")

barrier = threading.Barrier(3, action=on_barrier)
```

---

## 死锁

### 什么是死锁

```python
import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def task1():
    with lock1:
        print("Task1 acquired lock1")
        with lock2:  # 等待 lock2
            print("Task1 acquired lock2")

def task2():
    with lock2:
        print("Task2 acquired lock2")
        with lock1:  # 等待 lock1
            print("Task2 acquired lock1")

# 可能死锁！task1 持有 lock1 等 lock2，task2 持有 lock2 等 lock1
```

### 预防死锁

```python
import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

# 方法 1：固定锁顺序
def task1():
    with lock1:
        with lock2:
            pass

def task2():
    with lock1:  # 相同顺序
        with lock2:
            pass

# 方法 2：使用超时
def task_with_timeout():
    if lock1.acquire(timeout=1):
        try:
            if lock2.acquire(timeout=1):
                try:
                    pass
                finally:
                    lock2.release()
        finally:
            lock1.release()

# 方法 3：使用 RLock 避免自我死锁
rlock = threading.RLock()
```

---

## 线程安全的数据结构

### 使用锁保护

```python
import threading

class ThreadSafeList:
    def __init__(self):
        self._list = []
        self._lock = threading.Lock()

    def append(self, item):
        with self._lock:
            self._list.append(item)

    def pop(self):
        with self._lock:
            return self._list.pop() if self._list else None

    def __len__(self):
        with self._lock:
            return len(self._list)
```

### 使用 queue.Queue

```python
from queue import Queue

# 线程安全的队列
q = Queue()
q.put(item)     # 放入
item = q.get()  # 取出

# 详见下一章
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记释放锁 | 其他线程永久等待 | 使用 with 语句 |
| 锁顺序不一致 | 死锁 | 统一获取顺序 |
| 条件用 if | 虚假唤醒 | 用 while 循环检查 |
| 忘记 notify | 线程永久等待 | 确保通知 |
| 过度使用锁 | 性能下降 | 缩小临界区 |

---

## 小结

| 原语 | 用途 | 典型场景 |
|------|------|---------|
| Lock | 互斥访问 | 保护共享数据 |
| RLock | 可重入互斥 | 递归调用 |
| Condition | 条件等待 | 生产者-消费者 |
| Event | 简单信号 | 一次性通知 |
| Semaphore | 限制并发 | 连接池 |
| Barrier | 同步点 | 分阶段任务 |

