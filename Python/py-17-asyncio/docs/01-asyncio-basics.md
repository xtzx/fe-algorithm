# asyncio 基础

> async/await 语法、事件循环、协程与任务

## 1. 什么是 asyncio

asyncio 是 Python 的异步 I/O 框架，用于编写单线程并发代码。

```
传统同步:        异步并发:
────────         ────────
请求1 [===]      请求1 [=]    [=]
请求2    [===]   请求2   [=]  [=]
请求3       [===]请求3     [=][=]
─────────────    ─────────────
总时间: 9s       总时间: 3s
```

## 2. async/await 语法

### 定义异步函数

```python
async def fetch_data():
    # 异步操作
    await asyncio.sleep(1)
    return {"data": "result"}
```

### 调用异步函数

```python
# 方式 1: asyncio.run() (推荐)
result = asyncio.run(fetch_data())

# 方式 2: 在异步函数中 await
async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

## 3. 事件循环

事件循环是 asyncio 的核心，负责调度和执行协程。

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# asyncio.run() 会:
# 1. 创建事件循环
# 2. 运行协程
# 3. 关闭循环
asyncio.run(main())
```

### 获取当前循环

```python
async def get_loop():
    loop = asyncio.get_running_loop()
    print(f"Running: {loop.is_running()}")
```

## 4. 协程 vs 任务

### 协程 (Coroutine)

```python
async def fetch():
    await asyncio.sleep(1)
    return "done"

# 创建协程对象（不执行）
coro = fetch()

# await 时才执行
result = await coro
```

### 任务 (Task)

```python
# 创建任务（立即开始执行）
task = asyncio.create_task(fetch())

# 任务在后台执行，可以做其他事情
await asyncio.sleep(0.5)

# 获取结果
result = await task
```

### 关键区别

| 特性 | 协程 | 任务 |
|------|------|------|
| 执行时机 | await 时 | 创建时 |
| 并发执行 | 需要 gather | 自动并发 |
| 取消 | 不支持 | 支持 |

## 5. 并发执行

### 串行执行（慢）

```python
async def main():
    result1 = await fetch(1)  # 等待完成
    result2 = await fetch(2)  # 再等待
    # 总时间 = fetch(1) + fetch(2)
```

### 并发执行（快）

```python
async def main():
    # 同时启动
    task1 = asyncio.create_task(fetch(1))
    task2 = asyncio.create_task(fetch(2))

    # 等待所有完成
    result1 = await task1
    result2 = await task2
    # 总时间 = max(fetch(1), fetch(2))
```

## 6. async with 和 async for

### 异步上下文管理器

```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

async def main():
    async with AsyncResource() as resource:
        await resource.do_work()
```

### 异步迭代器

```python
async def async_generator(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for value in async_generator(5):
        print(value)
```

## 7. 与 JS 对比

```python
# Python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "result"}

result = asyncio.run(fetch_data())
```

```javascript
// JavaScript
async function fetchData() {
    await new Promise(r => setTimeout(r, 1000));
    return { data: "result" };
}

const result = await fetchData();
```

## 8. 常见错误

### ❌ 忘记 await

```python
async def main():
    result = fetch_data()  # 返回协程对象，不是结果！
```

### ✅ 正确

```python
async def main():
    result = await fetch_data()  # 返回实际结果
```

### ❌ 在同步函数中调用异步函数

```python
def sync_function():
    result = asyncio.run(fetch_data())  # 可以，但不推荐嵌套
```

### ✅ 推荐

```python
async def async_function():
    result = await fetch_data()
```

## 小结

| 概念 | 要点 |
|------|------|
| async def | 定义异步函数 |
| await | 等待异步操作完成 |
| asyncio.run() | 运行顶层协程 |
| create_task() | 创建并发任务 |
| async with | 异步资源管理 |
| async for | 异步迭代 |

