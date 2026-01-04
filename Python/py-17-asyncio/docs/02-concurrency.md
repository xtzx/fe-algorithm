# 并发原语

> gather、wait、create_task、TaskGroup

## 1. asyncio.gather()

并发执行多个协程，等待全部完成。

```python
async def main():
    results = await asyncio.gather(
        fetch(1),
        fetch(2),
        fetch(3),
    )
    # results = [result1, result2, result3]
```

### 处理异常

```python
# 默认：一个失败，抛出异常，其他继续运行
try:
    results = await asyncio.gather(fetch(1), bad_fetch(2))
except Exception as e:
    print(f"Error: {e}")

# return_exceptions=True：收集异常而不抛出
results = await asyncio.gather(
    fetch(1),
    bad_fetch(2),
    return_exceptions=True,
)
# results = [result1, Exception(...)]
```

## 2. asyncio.wait()

更灵活的等待，返回 done 和 pending 集合。

```python
tasks = [
    asyncio.create_task(fetch(1)),
    asyncio.create_task(fetch(2)),
]

done, pending = await asyncio.wait(tasks)
```

### 等待模式

```python
# 等待全部完成（默认）
done, pending = await asyncio.wait(tasks)

# 等待第一个完成
done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.FIRST_COMPLETED,
)

# 等待第一个异常
done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.FIRST_EXCEPTION,
)

# 带超时
done, pending = await asyncio.wait(
    tasks,
    timeout=5.0,
)
```

### 处理结果

```python
for task in done:
    if task.exception():
        print(f"Task failed: {task.exception()}")
    else:
        print(f"Task result: {task.result()}")

# 取消未完成的任务
for task in pending:
    task.cancel()
```

## 3. asyncio.create_task()

创建任务，立即开始执行。

```python
async def main():
    # 创建任务（开始执行）
    task = asyncio.create_task(fetch(1), name="fetch_1")

    # 做其他事情
    await asyncio.sleep(0.5)

    # 获取结果
    result = await task
```

### 任务属性

```python
task = asyncio.create_task(fetch(1), name="my_task")

task.get_name()     # "my_task"
task.done()         # True/False
task.cancelled()    # True/False
task.result()       # 结果（任务完成后）
task.exception()    # 异常（如果有）
```

### 添加回调

```python
def on_done(task):
    print(f"Task {task.get_name()} done")

task.add_done_callback(on_done)
```

## 4. TaskGroup (Python 3.11+)

结构化并发：自动管理任务生命周期。

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch(1))
        task2 = tg.create_task(fetch(2))

    # 退出时所有任务已完成
    print(task1.result())
    print(task2.result())
```

### TaskGroup 优势

1. **自动等待**：退出时自动等待所有任务完成
2. **异常传播**：一个失败，其他自动取消
3. **异常收集**：多个异常作为 ExceptionGroup 抛出

### 错误处理

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(good_task())
        tg.create_task(bad_task())  # 会失败
except ExceptionGroup as eg:
    for exc in eg.exceptions:
        print(f"Error: {exc}")
```

## 5. gather vs wait vs TaskGroup

| 特性 | gather | wait | TaskGroup |
|------|--------|------|-----------|
| 返回值 | 结果列表 | done/pending 集合 | 无 |
| 异常处理 | 抛出或收集 | 在任务中 | ExceptionGroup |
| 取消传播 | 不自动 | 不自动 | 自动 |
| 超时 | 需要包装 | 内置 | 需要包装 |
| Python 版本 | 3.4+ | 3.4+ | 3.11+ |

## 6. 与 JS 对比

```python
# Python
results = await asyncio.gather(fetch(1), fetch(2))
```

```javascript
// JavaScript
const results = await Promise.all([fetch(1), fetch(2)]);
```

```python
# Python - 等待第一个完成
done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.FIRST_COMPLETED,
)
```

```javascript
// JavaScript - 等待第一个完成
const result = await Promise.race([fetch(1), fetch(2)]);
```

## 7. 最佳实践

### 使用 TaskGroup（Python 3.11+）

```python
# ✅ 推荐
async with asyncio.TaskGroup() as tg:
    for item in items:
        tg.create_task(process(item))
```

### 限制并发数

```python
# 使用 Semaphore 限制并发
semaphore = asyncio.Semaphore(10)

async def limited_task(item):
    async with semaphore:
        return await process(item)

await asyncio.gather(*[limited_task(i) for i in items])
```

### 处理取消

```python
tasks = [asyncio.create_task(fetch(i)) for i in range(10)]

try:
    done, pending = await asyncio.wait(tasks, timeout=5.0)
finally:
    # 取消未完成的任务
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

## 小结

| 原语 | 用途 | 场景 |
|------|------|------|
| gather | 并发执行，收集结果 | 批量请求 |
| wait | 灵活等待策略 | 竞争、超时 |
| create_task | 后台任务 | 并行处理 |
| TaskGroup | 结构化并发 | 复杂任务组 |

