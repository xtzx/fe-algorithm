#!/usr/bin/env python3
"""
asyncio 基础示例
"""

import asyncio
import time


async def example_basic():
    """基础 async/await 示例"""
    print("=" * 60)
    print("1. 基础 async/await")
    print("=" * 60)

    async def fetch_data(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id, "data": f"Data for {item_id}"}

    # 单个异步调用
    result = await fetch_data(1)
    print(f"Single fetch: {result}")

    print()


async def example_concurrent():
    """并发执行示例"""
    print("=" * 60)
    print("2. 并发执行 vs 串行执行")
    print("=" * 60)

    async def fetch(item_id: int) -> dict:
        await asyncio.sleep(0.2)
        return {"id": item_id}

    # 串行执行
    start = time.perf_counter()
    result1 = await fetch(1)
    result2 = await fetch(2)
    result3 = await fetch(3)
    serial_time = time.perf_counter() - start
    print(f"串行执行: {serial_time:.2f}s")

    # 并发执行
    start = time.perf_counter()
    results = await asyncio.gather(fetch(1), fetch(2), fetch(3))
    concurrent_time = time.perf_counter() - start
    print(f"并发执行: {concurrent_time:.2f}s")
    print(f"加速比: {serial_time / concurrent_time:.1f}x")

    print()


async def example_create_task():
    """create_task 示例"""
    print("=" * 60)
    print("3. create_task 后台执行")
    print("=" * 60)

    async def background_task():
        for i in range(3):
            await asyncio.sleep(0.1)
            print(f"  Background: {i}")
        return "Background done"

    # 创建任务（立即开始执行）
    task = asyncio.create_task(background_task(), name="background")
    print(f"Task created: {task.get_name()}")

    # 主任务继续执行
    for i in range(2):
        await asyncio.sleep(0.15)
        print(f"  Main: {i}")

    # 等待后台任务完成
    result = await task
    print(f"Task result: {result}")

    print()


async def example_timeout():
    """超时控制示例"""
    print("=" * 60)
    print("4. 超时控制")
    print("=" * 60)

    async def slow_operation() -> str:
        await asyncio.sleep(2.0)
        return "done"

    # 使用 asyncio.timeout
    try:
        async with asyncio.timeout(0.5):
            result = await slow_operation()
            print(f"Result: {result}")
    except TimeoutError:
        print("Operation timed out (expected)")

    print()


async def example_cancel():
    """取消任务示例"""
    print("=" * 60)
    print("5. 任务取消")
    print("=" * 60)

    async def long_task():
        try:
            print("  Task started")
            await asyncio.sleep(10.0)
            print("  Task completed")
        except asyncio.CancelledError:
            print("  Task cancelled, cleaning up...")
            raise

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0.1)

    print("Cancelling task...")
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Task cancellation confirmed")

    print()


async def example_taskgroup():
    """TaskGroup 示例"""
    print("=" * 60)
    print("6. TaskGroup (Python 3.11+)")
    print("=" * 60)

    async def fetch(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id}

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch(1))
        task2 = tg.create_task(fetch(2))
        task3 = tg.create_task(fetch(3))

    print(f"Results: {task1.result()}, {task2.result()}, {task3.result()}")

    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  asyncio 基础示例")
    print("=" * 60 + "\n")

    await example_basic()
    await example_concurrent()
    await example_create_task()
    await example_timeout()
    await example_cancel()
    await example_taskgroup()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

