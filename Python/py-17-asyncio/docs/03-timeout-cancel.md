# 超时与取消

> asyncio.timeout()、wait_for()、任务取消、清理

## 1. asyncio.timeout() (Python 3.11+)

推荐的超时控制方式。

```python
async def main():
    try:
        async with asyncio.timeout(5.0):
            result = await slow_operation()
    except TimeoutError:
        print("Operation timed out")
```

### 动态调整超时

```python
async with asyncio.timeout(10.0) as cm:
    result1 = await phase1()

    # 延长超时（从现在开始再等 5 秒）
    cm.reschedule(asyncio.get_running_loop().time() + 5.0)

    result2 = await phase2()
```

### 检查是否超时

```python
async with asyncio.timeout(5.0) as cm:
    await operation()

# 超时后 cm.expired() 返回 True
if cm.expired():
    print("Operation was cut short by timeout")
```

## 2. asyncio.wait_for()

更简洁的超时控制（适用于单个协程）。

```python
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except TimeoutError:
    print("Operation timed out")
```

### timeout() vs wait_for()

| 特性 | timeout() | wait_for() |
|------|-----------|------------|
| 语法 | 上下文管理器 | 函数 |
| 多操作 | ✓ | ✗ |
| 动态调整 | ✓ | ✗ |
| 推荐 | 复杂场景 | 简单场景 |

## 3. 任务取消

### 基本取消

```python
task = asyncio.create_task(long_operation())

# 等待一会儿
await asyncio.sleep(1.0)

# 取消任务
task.cancel()

# 等待任务结束
try:
    await task
except asyncio.CancelledError:
    print("Task was cancelled")
```

### 带消息的取消

```python
task.cancel("User requested cancellation")

try:
    await task
except asyncio.CancelledError as e:
    print(f"Cancelled: {e.args}")
```

### 检查取消状态

```python
print(task.cancelled())  # True/False
print(task.done())       # True if cancelled or completed
```

## 4. 取消时的清理

### try/except 模式

```python
async def task_with_cleanup():
    resource = await acquire_resource()
    try:
        await do_work()
    except asyncio.CancelledError:
        print("Cleaning up...")
        await release_resource(resource)
        raise  # 必须重新抛出！
```

### finally 模式

```python
async def task_with_cleanup():
    resource = await acquire_resource()
    try:
        await do_work()
    finally:
        # 无论如何都会执行
        await release_resource(resource)
```

### 上下文管理器模式（推荐）

```python
async def task_with_cleanup():
    async with resource_manager() as resource:
        await do_work()
    # 自动清理
```

## 5. asyncio.shield()

保护关键操作不被取消。

```python
async def main():
    try:
        # shield 保护 critical_save 不被取消
        result = await asyncio.shield(critical_save())
    except asyncio.CancelledError:
        print("Main cancelled, but save continues")
        # 注意：critical_save() 仍在运行！
```

### shield 注意事项

```python
# ⚠️ 被取消后，shielded 任务仍在运行
# 需要等待它完成

inner_task = asyncio.create_task(critical_operation())

try:
    await asyncio.shield(inner_task)
except asyncio.CancelledError:
    # 等待内部任务完成
    await inner_task
```

## 6. 超时模式

### 带重试的超时

```python
async def retry_with_timeout(coro_factory, timeout, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout):
                return await coro_factory()
        except TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
    raise TimeoutError("All retries failed")
```

### 超时竞争

```python
async def fetch_fastest():
    tasks = [
        asyncio.create_task(fetch_from_source_a()),
        asyncio.create_task(fetch_from_source_b()),
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 取消其他任务
    for task in pending:
        task.cancel()

    return list(done)[0].result()
```

### 带默认值的超时

```python
async def with_timeout(coro, timeout, default=None):
    try:
        async with asyncio.timeout(timeout):
            return await coro
    except TimeoutError:
        return default

result = await with_timeout(slow_fetch(), 5.0, default={})
```

## 7. 与 JS 对比

```python
# Python
async with asyncio.timeout(5.0):
    result = await fetch()
```

```javascript
// JavaScript
const result = await Promise.race([
    fetch(),
    new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Timeout')), 5000)
    ),
]);
```

```python
# Python - 取消
task.cancel()
```

```javascript
// JavaScript - 取消
const controller = new AbortController();
const { signal } = controller;

fetch(url, { signal });
controller.abort();
```

## 8. 最佳实践

### 始终处理取消

```python
async def my_task():
    try:
        await do_work()
    except asyncio.CancelledError:
        await cleanup()
        raise  # 必须重新抛出
```

### 清理未完成的任务

```python
async def main():
    tasks = [asyncio.create_task(work(i)) for i in range(10)]

    try:
        async with asyncio.timeout(5.0):
            await asyncio.gather(*tasks)
    except TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
        # 等待取消完成
        await asyncio.gather(*tasks, return_exceptions=True)
```

### 使用 TaskGroup 自动取消

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task1())
        tg.create_task(task2())
except ExceptionGroup as eg:
    # 所有任务自动取消
    pass
```

## 小结

| 概念 | 用途 |
|------|------|
| timeout() | 超时控制（多操作） |
| wait_for() | 超时控制（单操作） |
| task.cancel() | 取消任务 |
| CancelledError | 取消异常 |
| shield() | 保护关键操作 |
| finally | 清理资源 |

