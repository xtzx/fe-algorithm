# 面试题

## 1. asyncio 和多线程的区别？

**答案**：

| 特性 | asyncio | 多线程 |
|------|---------|--------|
| 并发模型 | 协作式（单线程） | 抢占式（多线程） |
| 上下文切换 | 显式（await） | 隐式（OS调度） |
| 共享状态 | 安全（同线程） | 需要锁 |
| 适用场景 | I/O 密集型 | CPU 密集型 |
| GIL 影响 | 无 | 有 |
| 内存开销 | 低（协程轻量） | 高（线程堆栈） |

**何时选择 asyncio**：
- 大量 I/O 操作（网络请求、文件）
- 需要高并发但不是 CPU 密集
- 避免线程同步复杂性

**何时选择多线程**：
- CPU 密集型任务（配合多进程）
- 调用阻塞的同步库

---

## 2. 什么是事件循环？

**答案**：

事件循环是 asyncio 的核心调度器，负责：
1. 管理和调度协程
2. 处理 I/O 事件
3. 执行回调

```python
# 简化的事件循环工作原理
while True:
    # 1. 获取就绪的任务
    ready_tasks = get_ready_tasks()

    # 2. 执行就绪任务直到下一个 await
    for task in ready_tasks:
        task.run_until_await()

    # 3. 等待 I/O 事件
    wait_for_io_events()
```

**关键点**：
- 单线程运行
- 协程遇到 await 时让出控制权
- 事件循环决定下一个执行的协程

---

## 3. 如何限制并发数量？

**答案**：

**方式 1：Semaphore**
```python
semaphore = asyncio.Semaphore(10)

async def limited_task():
    async with semaphore:
        return await do_work()

await asyncio.gather(*[limited_task() for _ in range(100)])
```

**方式 2：TaskGroup + 分批**
```python
async def batch_process(items, batch_size=10):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        await asyncio.gather(*[process(item) for item in batch])
```

**方式 3：Queue + 固定工作者**
```python
async def worker(queue):
    while True:
        item = await queue.get()
        await process(item)

# 创建固定数量的工作者
workers = [asyncio.create_task(worker(queue)) for _ in range(10)]
```

---

## 4. 如何正确取消异步任务？

**答案**：

```python
# 1. 创建任务
task = asyncio.create_task(long_operation())

# 2. 取消任务
task.cancel()

# 3. 等待取消完成
try:
    await task
except asyncio.CancelledError:
    print("Task cancelled")

# 4. 在任务内部处理取消
async def long_operation():
    try:
        await do_work()
    except asyncio.CancelledError:
        await cleanup()  # 清理资源
        raise  # 必须重新抛出！
```

**关键点**：
- `cancel()` 不会立即停止任务
- 需要 await 任务等待取消完成
- 任务内必须重新抛出 CancelledError

---

## 5. gather 和 wait 的区别？

**答案**：

| 特性 | gather | wait |
|------|--------|------|
| 返回值 | 结果列表 | (done, pending) 集合 |
| 顺序 | 保持输入顺序 | 无序 |
| 异常 | 抛出或收集 | 在任务对象中 |
| 超时 | 需要包装 | 内置支持 |
| 部分等待 | 不支持 | FIRST_COMPLETED |

```python
# gather: 等待全部，返回结果列表
results = await asyncio.gather(task1, task2, task3)

# wait: 返回 done/pending 集合
done, pending = await asyncio.wait(
    [task1, task2, task3],
    return_when=asyncio.FIRST_COMPLETED,
)
```

---

## 6. TaskGroup 的优势是什么？

**答案**：

**优势**：
1. **结构化并发**：保证所有子任务在退出前完成
2. **自动取消**：一个任务失败，其他自动取消
3. **异常收集**：多个异常作为 ExceptionGroup

```python
# 无 TaskGroup：任务可能逃逸
tasks = [asyncio.create_task(work(i)) for i in range(10)]
await asyncio.gather(*tasks)
# 如果 gather 被取消，任务可能继续运行

# 有 TaskGroup：保证清理
async with asyncio.TaskGroup() as tg:
    for i in range(10):
        tg.create_task(work(i))
# 保证：所有任务完成或取消后才退出
```

---

## 7. 如何处理异步中的超时？

**答案**：

**方式 1：asyncio.timeout() (Python 3.11+)**
```python
async with asyncio.timeout(5.0):
    result = await slow_operation()
```

**方式 2：asyncio.wait_for()**
```python
result = await asyncio.wait_for(slow_operation(), timeout=5.0)
```

**方式 3：asyncio.wait() + timeout**
```python
done, pending = await asyncio.wait(tasks, timeout=5.0)
for task in pending:
    task.cancel()
```

**处理超时异常**：
```python
try:
    async with asyncio.timeout(5.0):
        await operation()
except TimeoutError:
    print("Operation timed out")
```

---

## 8. 异步函数可以调用同步函数吗？

**答案**：

**可以，但有注意事项**：

```python
# ✅ 可以直接调用快速的同步函数
async def async_func():
    result = sync_func()  # OK，如果很快
    return result

# ⚠️ 阻塞的同步函数会阻塞事件循环
async def bad_example():
    time.sleep(5)  # 阻塞整个事件循环！

# ✅ 使用 run_in_executor 运行阻塞操作
async def good_example():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_func)
    return result

# ✅ 使用 asyncio.to_thread (Python 3.9+)
async def better_example():
    result = await asyncio.to_thread(blocking_func)
    return result
```

---

## 9. 如何调试异步代码？

**答案**：

**1. 启用调试模式**
```python
asyncio.run(main(), debug=True)
# 或
PYTHONASYNCIODEBUG=1 python script.py
```

**2. 使用日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**3. 检测未 await 的协程**
```python
# 调试模式会警告未 await 的协程
# RuntimeWarning: coroutine 'xxx' was never awaited
```

**4. 使用 asyncio.current_task()**
```python
async def debug_info():
    task = asyncio.current_task()
    print(f"Current task: {task.get_name()}")
```

**5. IDE 调试**
- VS Code 支持异步断点
- PyCharm 支持协程调试

---

## 10. async for 和 async with 是什么？

**答案**：

**async with - 异步上下文管理器**
```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

async with AsyncResource() as resource:
    await resource.do_work()
```

**async for - 异步迭代器**
```python
class AsyncRange:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= self.n:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        value = self.i
        self.i += 1
        return value

async for i in AsyncRange(5):
    print(i)
```

**异步生成器**
```python
async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async for i in async_range(5):
    print(i)
```

---

## 11. 如何实现请求速率限制？

**答案**：

```python
class RateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = 1.0
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens += (now - self.last_update) * self.rate
            self.tokens = min(1.0, self.tokens)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait)
            self.tokens = 0
```

---

## 12. 协程和任务的区别？

**答案**：

| 特性 | 协程 (Coroutine) | 任务 (Task) |
|------|------------------|-------------|
| 创建 | `async def` 调用 | `create_task()` |
| 执行时机 | await 时 | 创建时 |
| 可取消 | 否 | 是 |
| 有名字 | 否 | 是 |
| 获取结果 | await | await 或 result() |

```python
# 协程：惰性，await 时执行
coro = fetch_data()  # 不执行
result = await coro  # 执行

# 任务：立即开始执行
task = asyncio.create_task(fetch_data())  # 开始执行
result = await task  # 获取结果
```

