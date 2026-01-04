# 同步原语

> Lock、Semaphore、Event、Condition、Queue

## 1. Lock

保护共享状态，防止竞态条件。

```python
lock = asyncio.Lock()

async def safe_increment():
    async with lock:
        # 临界区：同一时间只有一个协程能进入
        value = await get_value()
        await set_value(value + 1)
```

### Lock 示例

```python
class Counter:
    def __init__(self):
        self.value = 0
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            self.value += 1
            return self.value
```

### 注意：Lock 是协程级别的

```python
# asyncio.Lock 保护协程并发，不是线程
# 如果需要线程安全，使用 threading.Lock
```

## 2. Semaphore

限制并发数量。

```python
semaphore = asyncio.Semaphore(10)  # 最多 10 个并发

async def limited_task():
    async with semaphore:
        # 最多 10 个协程同时在这里
        await do_work()
```

### 限流示例

```python
async def fetch_all(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url):
        async with semaphore:
            return await httpx.get(url)

    return await asyncio.gather(*[fetch_one(url) for url in urls])
```

### BoundedSemaphore

```python
# 检测 release 过多的错误
semaphore = asyncio.BoundedSemaphore(10)

# 如果 release 超过 acquire 次数，会抛出 ValueError
```

## 3. Event

协程间的信号通知。

```python
event = asyncio.Event()

async def waiter():
    print("Waiting for event...")
    await event.wait()  # 阻塞直到 event.set()
    print("Event received!")

async def setter():
    await asyncio.sleep(1)
    event.set()  # 通知所有等待者
```

### Event 方法

```python
event = asyncio.Event()

event.set()      # 设置事件
event.clear()    # 清除事件
event.is_set()   # 检查是否设置
await event.wait()  # 等待事件
```

### 一次性开关

```python
class Gate:
    def __init__(self):
        self._event = asyncio.Event()

    def open(self):
        self._event.set()

    async def wait(self):
        await self._event.wait()
```

## 4. Condition

复杂的同步条件。

```python
condition = asyncio.Condition()
queue = []

async def producer():
    async with condition:
        queue.append(item)
        condition.notify_all()  # 通知所有等待者

async def consumer():
    async with condition:
        while not queue:
            await condition.wait()  # 等待条件
        item = queue.pop(0)
```

### Condition 方法

```python
async with condition:
    await condition.wait()        # 等待通知
    await condition.wait_for(predicate)  # 等待条件为真
    condition.notify()            # 通知一个等待者
    condition.notify_all()        # 通知所有等待者
```

### 生产者/消费者示例

```python
async def producer():
    for i in range(10):
        async with condition:
            while len(queue) >= MAX_SIZE:
                await condition.wait()
            queue.append(i)
            condition.notify_all()

async def consumer():
    while True:
        async with condition:
            while not queue:
                await condition.wait()
            item = queue.pop(0)
            condition.notify_all()
```

## 5. Queue

异步队列，生产者/消费者最佳选择。

```python
queue = asyncio.Queue(maxsize=100)

async def producer():
    for i in range(10):
        await queue.put(i)  # 队列满时阻塞
    await queue.put(None)   # 结束信号

async def consumer():
    while True:
        item = await queue.get()  # 队列空时阻塞
        if item is None:
            break
        print(f"Got: {item}")
        queue.task_done()
```

### Queue 方法

```python
queue = asyncio.Queue(maxsize=10)

await queue.put(item)     # 放入（阻塞）
queue.put_nowait(item)    # 放入（非阻塞，满则抛异常）

item = await queue.get()  # 获取（阻塞）
item = queue.get_nowait() # 获取（非阻塞，空则抛异常）

queue.task_done()         # 标记任务完成
await queue.join()        # 等待所有任务完成

queue.qsize()             # 当前大小
queue.empty()             # 是否为空
queue.full()              # 是否已满
```

### 多消费者模式

```python
async def worker(queue, worker_id):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        await process(item)
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    # 启动多个工作者
    workers = [
        asyncio.create_task(worker(queue, i))
        for i in range(5)
    ]

    # 添加工作
    for item in items:
        await queue.put(item)

    # 发送停止信号
    for _ in range(5):
        await queue.put(None)

    await asyncio.gather(*workers)
```

## 6. Barrier (Python 3.11+)

等待所有参与者到达同步点。

```python
barrier = asyncio.Barrier(3)  # 3 个参与者

async def worker():
    print("Phase 1")
    await barrier.wait()  # 等待所有人到达
    print("Phase 2")
```

## 7. 与 JS 对比

Python 的同步原语比 JS 丰富得多：

```python
# Python - 多种同步原语
lock = asyncio.Lock()
semaphore = asyncio.Semaphore(10)
event = asyncio.Event()
queue = asyncio.Queue()
```

```javascript
// JavaScript - 需要自己实现或使用库
// 没有内置的 Lock、Semaphore 等
// 通常使用 Promise 和回调模式
```

## 8. 选择指南

| 场景 | 原语 |
|------|------|
| 保护共享状态 | Lock |
| 限制并发数 | Semaphore |
| 一次性通知 | Event |
| 复杂条件等待 | Condition |
| 任务队列 | Queue |
| 同步点 | Barrier |

## 小结

| 原语 | 用途 | 特点 |
|------|------|------|
| Lock | 互斥 | 同一时间一个协程 |
| Semaphore | 限流 | 限制并发数 |
| Event | 通知 | 一对多通知 |
| Condition | 条件等待 | 复杂同步 |
| Queue | 生产者/消费者 | 线程安全 |
| Barrier | 同步点 | 等待所有人 |

