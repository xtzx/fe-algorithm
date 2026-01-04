# 错误处理

> 异常收集、部分失败处理、结构化并发

## 1. 单任务异常

```python
async def main():
    try:
        result = await risky_operation()
    except ValueError as e:
        print(f"Value error: {e}")
    except asyncio.TimeoutError:
        print("Operation timed out")
    except asyncio.CancelledError:
        print("Operation was cancelled")
        raise  # 重新抛出取消
```

## 2. gather 异常处理

### 默认行为：抛出第一个异常

```python
try:
    results = await asyncio.gather(
        fetch(1),
        bad_fetch(2),  # 会抛出异常
        fetch(3),
    )
except Exception as e:
    print(f"One task failed: {e}")
    # 其他任务继续运行
```

### 收集异常

```python
results = await asyncio.gather(
    fetch(1),
    bad_fetch(2),
    fetch(3),
    return_exceptions=True,
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Task {i} failed: {result}")
    else:
        print(f"Task {i} succeeded: {result}")
```

## 3. TaskGroup 异常处理 (Python 3.11+)

### ExceptionGroup

TaskGroup 将多个异常打包为 ExceptionGroup：

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(good_task())
        tg.create_task(bad_task_1())  # ValueError
        tg.create_task(bad_task_2())  # TypeError
except ExceptionGroup as eg:
    print(f"ExceptionGroup: {eg}")
    for exc in eg.exceptions:
        print(f"  - {type(exc).__name__}: {exc}")
```

### except* 语法 (Python 3.11+)

```python
try:
    async with asyncio.TaskGroup() as tg:
        # ...
except* ValueError as eg:
    print(f"ValueError: {eg.exceptions}")
except* TypeError as eg:
    print(f"TypeError: {eg.exceptions}")
```

### 自动取消

TaskGroup 中一个任务失败，其他任务自动取消：

```python
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(long_task())    # 会被取消
    task2 = tg.create_task(failing_task()) # 失败
    task3 = tg.create_task(long_task())    # 会被取消
# 失败时，task1 和 task3 自动取消
```

## 4. 部分失败处理

### 继续处理成功的任务

```python
async def process_all(items):
    results = await asyncio.gather(
        *[process(item) for item in items],
        return_exceptions=True,
    )

    successes = []
    failures = []

    for item, result in zip(items, results):
        if isinstance(result, Exception):
            failures.append((item, result))
        else:
            successes.append((item, result))

    return successes, failures
```

### 使用 wait 处理

```python
tasks = [asyncio.create_task(process(item)) for item in items]

done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.ALL_COMPLETED,
)

for task in done:
    try:
        result = task.result()
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
```

## 5. 任务异常处理

### 添加异常回调

```python
def handle_task_exception(task):
    if task.cancelled():
        print("Task cancelled")
    elif task.exception():
        print(f"Task failed: {task.exception()}")

task = asyncio.create_task(risky_operation())
task.add_done_callback(handle_task_exception)
```

### 全局异常处理

```python
def exception_handler(loop, context):
    msg = context.get("exception", context["message"])
    print(f"Unhandled exception: {msg}")

loop = asyncio.get_running_loop()
loop.set_exception_handler(exception_handler)
```

## 6. 结构化并发

### 为什么需要结构化并发

```
非结构化:
  main ──┬── task1 (可能逃逸)
         ├── task2 (可能逃逸)
         └── task3 (可能逃逸)
  # 任务可能在 main 结束后继续运行

结构化:
  main ──┬── task1 ──┐
         ├── task2 ──┼── 全部完成后才退出
         └── task3 ──┘
  # 保证所有子任务在 main 前完成
```

### 使用 TaskGroup 实现结构化并发

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task1())
        tg.create_task(task2())
        tg.create_task(task3())
    # 保证: 所有任务完成或取消后才到这里
    print("All tasks done")
```

### 手动实现结构化并发

```python
async def structured_main():
    tasks = []
    try:
        tasks = [
            asyncio.create_task(task1()),
            asyncio.create_task(task2()),
        ]
        await asyncio.gather(*tasks)
    finally:
        # 确保所有任务完成
        for task in tasks:
            if not task.done():
                task.cancel()
        # 等待取消完成
        await asyncio.gather(*tasks, return_exceptions=True)
```

## 7. 错误恢复模式

### 重试失败的任务

```python
async def retry_on_failure(coro_factory, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
```

### 降级处理

```python
async def fetch_with_fallback(primary_url, fallback_url):
    try:
        return await fetch(primary_url)
    except Exception:
        return await fetch(fallback_url)
```

### 超时后重试

```python
async def fetch_with_timeout_retry(url, timeout=5.0, retries=3):
    for attempt in range(retries):
        try:
            async with asyncio.timeout(timeout):
                return await fetch(url)
        except TimeoutError:
            if attempt == retries - 1:
                raise
```

## 8. 最佳实践

### 始终处理 CancelledError

```python
async def my_task():
    try:
        await do_work()
    except asyncio.CancelledError:
        await cleanup()
        raise  # 必须重新抛出
```

### 使用 TaskGroup 进行结构化并发

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(task1())
    tg.create_task(task2())
```

### 收集并处理所有异常

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
errors = [r for r in results if isinstance(r, Exception)]
if errors:
    # 记录或处理错误
```

## 小结

| 模式 | 用途 |
|------|------|
| try/except | 单任务异常 |
| return_exceptions | 收集 gather 异常 |
| ExceptionGroup | TaskGroup 多异常 |
| except* | 分类处理异常 |
| 结构化并发 | 保证任务生命周期 |

