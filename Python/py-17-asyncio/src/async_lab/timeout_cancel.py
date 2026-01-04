"""
超时与取消

包含:
- asyncio.timeout()
- asyncio.wait_for()
- 任务取消
- 取消时的清理
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


# =============================================================================
# 1. asyncio.timeout() (Python 3.11+)
# =============================================================================


async def timeout_demo():
    """
    使用 asyncio.timeout() 控制超时
    """
    async def slow_operation() -> str:
        await asyncio.sleep(5.0)
        return "done"

    try:
        async with asyncio.timeout(1.0):
            result = await slow_operation()
            return result
    except TimeoutError:
        print("Operation timed out")
        return None


async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    default: T | None = None,
) -> T | None:
    """
    带超时的协程执行

    Example:
        ```python
        result = await with_timeout(slow_operation(), 5.0, default=None)
        ```
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            return await coro
    except TimeoutError:
        return default


async def timeout_reschedule_demo():
    """
    动态调整超时时间
    """
    async def variable_task() -> str:
        await asyncio.sleep(0.5)
        return "phase1"

    async with asyncio.timeout(2.0) as cm:
        # 第一阶段
        result1 = await variable_task()
        print(f"Phase 1: {result1}")

        # 延长超时（从现在开始再等 2 秒）
        cm.reschedule(asyncio.get_running_loop().time() + 2.0)

        # 第二阶段
        result2 = await variable_task()
        print(f"Phase 2: {result2}")

    return result1, result2


# =============================================================================
# 2. asyncio.wait_for()
# =============================================================================


async def wait_for_demo():
    """
    使用 asyncio.wait_for() 控制超时

    与 timeout() 类似，但更简洁
    """
    async def slow_operation() -> str:
        await asyncio.sleep(5.0)
        return "done"

    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
        return result
    except TimeoutError:
        print("Operation timed out")
        return None


# =============================================================================
# 3. 任务取消
# =============================================================================


async def cancel_task_demo():
    """
    取消任务示例
    """
    async def long_running_task() -> str:
        try:
            print("Task started")
            await asyncio.sleep(10.0)
            print("Task completed")
            return "done"
        except asyncio.CancelledError:
            print("Task was cancelled")
            raise

    # 创建任务
    task = asyncio.create_task(long_running_task())

    # 等待一会儿
    await asyncio.sleep(0.5)

    # 取消任务
    task.cancel()

    # 等待任务结束（会抛出 CancelledError）
    try:
        await task
    except asyncio.CancelledError:
        print("Task cancellation confirmed")

    print(f"Task cancelled: {task.cancelled()}")
    return task.cancelled()


async def cancel_with_message():
    """
    带消息的取消（Python 3.9+）
    """
    async def task_with_cancel_message():
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError as e:
            print(f"Cancelled with message: {e.args}")
            raise

    task = asyncio.create_task(task_with_cancel_message())
    await asyncio.sleep(0.1)

    # 带消息取消
    task.cancel("User requested cancellation")

    try:
        await task
    except asyncio.CancelledError:
        pass


# =============================================================================
# 4. 取消时的清理
# =============================================================================


async def cleanup_on_cancel():
    """
    取消时执行清理操作
    """
    resource_acquired = False

    async def task_with_cleanup():
        nonlocal resource_acquired

        # 获取资源
        resource_acquired = True
        print("Resource acquired")

        try:
            await asyncio.sleep(10.0)
            return "done"
        except asyncio.CancelledError:
            print("Cancellation received, cleaning up...")
            # 清理资源
            await asyncio.sleep(0.1)  # 模拟清理
            resource_acquired = False
            print("Cleanup completed")
            raise  # 必须重新抛出

    task = asyncio.create_task(task_with_cleanup())
    await asyncio.sleep(0.1)

    print(f"Before cancel: resource_acquired={resource_acquired}")
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    print(f"After cancel: resource_acquired={resource_acquired}")
    return not resource_acquired


@asynccontextmanager
async def cleanup_context():
    """
    使用上下文管理器处理清理

    Example:
        ```python
        async with cleanup_context():
            await some_operation()
        # 即使被取消，也会执行清理
        ```
    """
    print("Entering context")
    try:
        yield
    except asyncio.CancelledError:
        print("Cancelled in context, cleaning up...")
        await asyncio.sleep(0.1)  # 模拟清理
        print("Cleanup done")
        raise
    finally:
        print("Exiting context")


async def shield_demo():
    """
    使用 shield 保护关键操作不被取消
    """
    async def critical_operation() -> str:
        print("Critical operation started")
        await asyncio.sleep(0.3)
        print("Critical operation completed")
        return "critical_done"

    async def main_task() -> str | None:
        try:
            # shield 保护 critical_operation 不被取消
            result = await asyncio.shield(critical_operation())
            return result
        except asyncio.CancelledError:
            print("Main task cancelled, but critical operation continues")
            raise

    task = asyncio.create_task(main_task())
    await asyncio.sleep(0.1)

    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        # 等待关键操作完成
        await asyncio.sleep(0.3)

    return True


# =============================================================================
# 5. 超时模式
# =============================================================================


async def retry_with_timeout(
    coro_factory,
    timeout: float,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> Any:
    """
    带超时的重试

    Args:
        coro_factory: 返回协程的函数（每次重试需要新协程）
        timeout: 每次尝试的超时时间
        max_retries: 最大重试次数
        retry_delay: 重试间隔
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout):
                return await coro_factory()
        except TimeoutError as e:
            last_error = e
            print(f"Attempt {attempt + 1} timed out")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

    raise last_error or TimeoutError("All retries failed")


async def timeout_race():
    """
    超时竞争：在多个源中获取数据，返回最快的
    """
    async def fetch_from_source(source: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Data from {source}"

    tasks = [
        asyncio.create_task(fetch_from_source("source_a", 0.3)),
        asyncio.create_task(fetch_from_source("source_b", 0.1)),
        asyncio.create_task(fetch_from_source("source_c", 0.2)),
    ]

    # 等待第一个完成
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 取消其他任务
    for task in pending:
        task.cancel()

    result = list(done)[0].result()
    print(f"Winner: {result}")

    # 等待取消完成
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            pass

    return result

