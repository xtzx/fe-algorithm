"""
并发原语

包含:
- asyncio.gather()
- asyncio.wait()
- asyncio.create_task()
- TaskGroup (Python 3.11+)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Sequence, TypeVar

T = TypeVar("T")


# =============================================================================
# 1. asyncio.gather()
# =============================================================================


async def run_concurrently(*coros: Coroutine[Any, Any, T]) -> list[T]:
    """
    并发执行多个协程

    Example:
        ```python
        results = await run_concurrently(
            fetch(1),
            fetch(2),
            fetch(3),
        )
        ```
    """
    return await asyncio.gather(*coros)


async def gather_with_errors(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = True,
) -> list[T | Exception]:
    """
    并发执行，收集错误而不抛出

    Example:
        ```python
        results = await gather_with_errors(
            fetch(1),
            failing_fetch(2),  # 这个会失败
            fetch(3),
        )
        # results = [result1, Exception(...), result3]
        ```
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def gather_demo():
    """
    gather 使用示例
    """
    async def fetch(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id}

    # 并发获取多个数据
    results = await asyncio.gather(
        fetch(1),
        fetch(2),
        fetch(3),
    )

    print(f"Results: {results}")
    return results


# =============================================================================
# 2. asyncio.wait()
# =============================================================================


async def wait_demo():
    """
    asyncio.wait() 使用示例

    与 gather 不同，wait 返回 done 和 pending 两个集合
    """
    async def fetch(item_id: int, delay: float) -> dict:
        await asyncio.sleep(delay)
        return {"id": item_id}

    tasks = [
        asyncio.create_task(fetch(1, 0.1)),
        asyncio.create_task(fetch(2, 0.2)),
        asyncio.create_task(fetch(3, 0.3)),
    ]

    # 等待所有完成
    done, pending = await asyncio.wait(tasks)
    print(f"Done: {len(done)}, Pending: {len(pending)}")

    return [t.result() for t in done]


async def wait_first_completed():
    """
    等待第一个完成
    """
    async def fetch(item_id: int, delay: float) -> dict:
        await asyncio.sleep(delay)
        return {"id": item_id}

    tasks = [
        asyncio.create_task(fetch(1, 0.3), name="slow"),
        asyncio.create_task(fetch(2, 0.1), name="fast"),
        asyncio.create_task(fetch(3, 0.2), name="medium"),
    ]

    # 等待第一个完成
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    first_done = list(done)[0]
    print(f"First completed: {first_done.get_name()}")
    print(f"Pending: {len(pending)}")

    # 取消剩余任务
    for task in pending:
        task.cancel()

    return first_done.result()


async def wait_with_timeout_demo():
    """
    带超时的等待
    """
    async def slow_fetch(item_id: int) -> dict:
        await asyncio.sleep(1.0)
        return {"id": item_id}

    tasks = [
        asyncio.create_task(slow_fetch(1)),
        asyncio.create_task(slow_fetch(2)),
    ]

    # 等待最多 0.5 秒
    done, pending = await asyncio.wait(
        tasks,
        timeout=0.5,
    )

    print(f"Done: {len(done)}, Pending: {len(pending)}")

    # 取消超时的任务
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    return list(done)


# =============================================================================
# 3. asyncio.create_task()
# =============================================================================


async def create_task_demo():
    """
    create_task 使用示例
    """
    async def background_task():
        for i in range(5):
            await asyncio.sleep(0.1)
            print(f"Background: {i}")
        return "Background done"

    # 创建后台任务
    task = asyncio.create_task(background_task(), name="background")

    # 做其他事情
    for i in range(3):
        await asyncio.sleep(0.15)
        print(f"Main: {i}")

    # 等待后台任务完成
    result = await task
    print(f"Task result: {result}")

    return result


async def fire_and_forget(coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """
    创建"即发即忘"的任务

    注意：应该保存任务引用，否则可能被垃圾回收
    """
    task = asyncio.create_task(coro)

    # 添加回调处理异常
    def handle_exception(t: asyncio.Task):
        if not t.cancelled() and t.exception():
            print(f"Task failed: {t.exception()}")

    task.add_done_callback(handle_exception)
    return task


# =============================================================================
# 4. TaskGroup (Python 3.11+)
# =============================================================================


async def task_group_demo():
    """
    TaskGroup 使用示例

    TaskGroup 提供结构化并发：
    - 自动等待所有任务完成
    - 一个任务失败，其他任务自动取消
    - 异常会被收集并作为 ExceptionGroup 抛出
    """
    async def fetch(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id}

    results = []

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch(1))
        task2 = tg.create_task(fetch(2))
        task3 = tg.create_task(fetch(3))

    # TaskGroup 退出后，所有任务都已完成
    results = [task1.result(), task2.result(), task3.result()]
    print(f"Results: {results}")

    return results


async def task_group_error_handling():
    """
    TaskGroup 错误处理示例
    """
    async def good_task(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id}

    async def bad_task() -> None:
        await asyncio.sleep(0.05)
        raise ValueError("Something went wrong")

    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(good_task(1))
            task2 = tg.create_task(bad_task())  # 这个会失败
            task3 = tg.create_task(good_task(3))

    except ExceptionGroup as eg:
        print(f"ExceptionGroup caught: {eg}")
        for exc in eg.exceptions:
            print(f"  - {type(exc).__name__}: {exc}")
        return None

    return [task1.result(), task3.result()]


async def task_group_dynamic():
    """
    动态添加任务到 TaskGroup
    """
    async def fetch(item_id: int) -> dict:
        await asyncio.sleep(0.1)
        return {"id": item_id}

    tasks: list[asyncio.Task] = []

    async with asyncio.TaskGroup() as tg:
        for i in range(5):
            task = tg.create_task(fetch(i))
            tasks.append(task)

    results = [t.result() for t in tasks]
    print(f"Results: {results}")

    return results


# =============================================================================
# 5. 便捷函数
# =============================================================================


async def map_concurrent(
    func: Callable[[T], Coroutine[Any, Any, Any]],
    items: Sequence[T],
    max_concurrent: int | None = None,
) -> list:
    """
    并发 map 函数

    Example:
        ```python
        async def fetch(url: str) -> dict:
            ...

        results = await map_concurrent(fetch, urls, max_concurrent=10)
        ```
    """
    if max_concurrent is None:
        # 不限制并发
        return await asyncio.gather(*[func(item) for item in items])

    # 使用 Semaphore 限制并发
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(item: T):
        async with semaphore:
            return await func(item)

    return await asyncio.gather(*[limited(item) for item in items])

