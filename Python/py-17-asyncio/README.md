# P17: asyncio 并发

> 掌握结构化并发、取消、超时、错误处理

## 🎯 学完后能做

- 编写高效的异步代码
- 正确处理取消和超时
- 使用 TaskGroup 管理任务

## 🚀 快速开始

```bash
# 进入项目目录
cd py-17-asyncio

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行示例
python examples/demo_basics.py
```

## 📁 目录结构

```
py-17-asyncio/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-asyncio-basics.md     # asyncio 基础
│   ├── 02-concurrency.md        # 并发原语
│   ├── 03-timeout-cancel.md     # 超时与取消
│   ├── 04-sync-primitives.md    # 同步原语
│   ├── 05-error-handling.md     # 错误处理
│   ├── 06-patterns.md           # 实战模式
│   ├── 07-exercises.md          # 练习题
│   └── 08-interview.md          # 面试题
├── src/async_lab/
│   ├── __init__.py
│   ├── basics.py                # asyncio 基础
│   ├── concurrency.py           # 并发原语
│   ├── timeout_cancel.py        # 超时与取消
│   ├── sync_primitives.py       # 同步原语
│   ├── patterns.py              # 实战模式
│   └── stats.py                 # 统计工具
├── tests/
│   ├── conftest.py
│   ├── test_basics.py
│   ├── test_concurrency.py
│   ├── test_timeout_cancel.py
│   └── test_patterns.py
├── examples/
│   ├── demo_basics.py
│   ├── demo_patterns.py
│   └── demo_producer_consumer.py
└── scripts/
    └── run_examples.sh
```

## 🆚 Python vs JavaScript 对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| 异步函数 | `async def` | `async function` |
| 等待 | `await` | `await` |
| 并发执行 | `asyncio.gather()` | `Promise.all()` |
| 事件循环 | `asyncio.run()` | 内置 |
| 任务取消 | `task.cancel()` | `AbortController` |
| 超时 | `asyncio.timeout()` | `Promise.race()` |

## 🔧 核心概念

### 1. async/await 基础

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "result"}

# 运行
result = asyncio.run(fetch_data())
```

### 2. 并发执行

```python
import asyncio

async def main():
    # 并发执行多个任务
    results = await asyncio.gather(
        fetch_data(1),
        fetch_data(2),
        fetch_data(3),
    )
    return results
```

### 3. TaskGroup (Python 3.11+)

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_data(1))
        task2 = tg.create_task(fetch_data(2))

    # 所有任务完成后才会继续
    print(task1.result(), task2.result())
```

### 4. 超时控制

```python
async def main():
    async with asyncio.timeout(5.0):
        result = await slow_operation()
```

### 5. 并发限制

```python
async def main():
    semaphore = asyncio.Semaphore(10)  # 最多 10 个并发

    async def limited_task():
        async with semaphore:
            await do_work()

    await asyncio.gather(*[limited_task() for _ in range(100)])
```

## 📚 学习路径

1. **asyncio 基础** - async/await、事件循环
2. **并发原语** - gather、wait、create_task、TaskGroup
3. **超时与取消** - timeout、wait_for、取消处理
4. **同步原语** - Lock、Semaphore、Event、Queue
5. **错误处理** - 异常收集、部分失败
6. **实战模式** - 并发请求、生产者/消费者

## ✅ 功能清单

- [x] async/await 语法
- [x] 事件循环
- [x] 协程 vs 任务
- [x] asyncio.gather()
- [x] asyncio.wait()
- [x] TaskGroup
- [x] 超时控制
- [x] 任务取消
- [x] Lock、Semaphore
- [x] Event、Queue
- [x] 并发请求模式
- [x] 生产者/消费者
- [x] 统计报表（p50/p95）

