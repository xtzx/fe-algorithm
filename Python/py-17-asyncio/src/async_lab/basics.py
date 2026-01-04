"""
asyncio 基础

包含:
- async/await 语法
- 事件循环
- 协程 vs 任务
- asyncio.run()
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


# =============================================================================
# 1. async/await 基础
# =============================================================================


async def sleep_demo(seconds: float) -> str:
    """
    简单的异步函数示例

    Example:
        ```python
        result = await sleep_demo(1.0)
        print(result)  # "Slept for 1.0 seconds"
        ```
    """
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"


async def fetch_data(item_id: int, delay: float = 0.1) -> dict[str, Any]:
    """
    模拟异步数据获取

    Example:
        ```python
        data = await fetch_data(123)
        print(data)  # {"id": 123, "status": "success"}
        ```
    """
    await asyncio.sleep(delay)
    return {"id": item_id, "status": "success"}


# =============================================================================
# 2. 运行异步代码
# =============================================================================


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    运行异步协程的便捷函数

    这是 asyncio.run() 的简单封装

    Example:
        ```python
        result = run_async(fetch_data(1))
        print(result)
        ```
    """
    return asyncio.run(coro)


# =============================================================================
# 3. 协程 vs 任务
# =============================================================================


async def coroutine_example():
    """
    协程示例

    协程是异步函数调用的结果，但不会立即执行
    """
    # 创建协程对象（不执行）
    coro = fetch_data(1)
    print(f"Created coroutine: {coro}")

    # await 时才执行
    result = await coro
    print(f"Result: {result}")

    return result


async def task_example():
    """
    任务示例

    任务是对协程的包装，会立即开始执行
    """
    # 创建任务（立即开始执行）
    task = asyncio.create_task(fetch_data(1))
    print(f"Created task: {task}")
    print(f"Task state: {task.done()}")  # False - 还在执行

    # 可以做其他事情...
    await asyncio.sleep(0.2)

    # 获取结果
    result = await task
    print(f"Task completed: {task.done()}")  # True
    print(f"Result: {result}")

    return result


async def coroutine_vs_task_demo():
    """
    展示协程和任务的区别

    - 协程：惰性执行，await 时才运行
    - 任务：创建后立即开始执行
    """
    print("=== 协程示例 ===")
    # 协程按顺序执行（串行）
    start = asyncio.get_event_loop().time()

    result1 = await fetch_data(1, 0.1)
    result2 = await fetch_data(2, 0.1)

    serial_time = asyncio.get_event_loop().time() - start
    print(f"串行执行: {serial_time:.2f}s")

    print("\n=== 任务示例 ===")
    # 任务并发执行
    start = asyncio.get_event_loop().time()

    task1 = asyncio.create_task(fetch_data(1, 0.1))
    task2 = asyncio.create_task(fetch_data(2, 0.1))

    result1 = await task1
    result2 = await task2

    concurrent_time = asyncio.get_event_loop().time() - start
    print(f"并发执行: {concurrent_time:.2f}s")
    print(f"加速比: {serial_time / concurrent_time:.1f}x")

    return result1, result2


# =============================================================================
# 4. 事件循环
# =============================================================================


def event_loop_demo():
    """
    事件循环示例

    事件循环是 asyncio 的核心，负责调度和执行协程
    """
    async def main():
        print("Main started")
        await asyncio.sleep(0.1)
        print("Main finished")
        return "done"

    # 方式 1: asyncio.run() (推荐)
    # 自动创建事件循环、运行协程、关闭循环
    result = asyncio.run(main())
    print(f"Result: {result}")

    return result


async def get_running_loop_demo():
    """
    获取当前事件循环
    """
    loop = asyncio.get_running_loop()
    print(f"Running loop: {loop}")
    print(f"Is running: {loop.is_running()}")
    return loop


# =============================================================================
# 5. 异步上下文管理器
# =============================================================================


class AsyncResource:
    """
    异步上下文管理器示例

    使用 async with 自动管理资源

    Example:
        ```python
        async with AsyncResource() as resource:
            await resource.do_work()
        ```
    """

    def __init__(self, name: str = "resource"):
        self.name = name
        self._connected = False

    async def __aenter__(self) -> "AsyncResource":
        """异步进入上下文"""
        print(f"Connecting to {self.name}...")
        await asyncio.sleep(0.1)  # 模拟连接
        self._connected = True
        print(f"Connected to {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步退出上下文"""
        print(f"Disconnecting from {self.name}...")
        await asyncio.sleep(0.05)  # 模拟断开连接
        self._connected = False
        print(f"Disconnected from {self.name}")

    async def do_work(self) -> str:
        """执行工作"""
        if not self._connected:
            raise RuntimeError("Not connected")
        await asyncio.sleep(0.1)
        return f"Work done with {self.name}"


# =============================================================================
# 6. 异步迭代器
# =============================================================================


class AsyncCounter:
    """
    异步迭代器示例

    使用 async for 遍历

    Example:
        ```python
        async for value in AsyncCounter(5):
            print(value)
        ```
    """

    def __init__(self, max_value: int, delay: float = 0.1):
        self.max_value = max_value
        self.delay = delay
        self._current = 0

    def __aiter__(self) -> "AsyncCounter":
        return self

    async def __anext__(self) -> int:
        if self._current >= self.max_value:
            raise StopAsyncIteration

        await asyncio.sleep(self.delay)
        value = self._current
        self._current += 1
        return value


async def async_generator_demo(max_value: int, delay: float = 0.1):
    """
    异步生成器示例

    使用 async for 遍历

    Example:
        ```python
        async for value in async_generator_demo(5):
            print(value)
        ```
    """
    for i in range(max_value):
        await asyncio.sleep(delay)
        yield i

