"""超时与取消测试"""

import asyncio

import pytest

from async_lab.timeout_cancel import with_timeout


class TestTimeout:
    """超时测试"""

    async def test_with_timeout_success(self) -> None:
        async def fast_operation() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await with_timeout(fast_operation(), timeout_seconds=1.0)
        assert result == "done"

    async def test_with_timeout_exceeded(self) -> None:
        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "done"

        result = await with_timeout(
            slow_operation(),
            timeout_seconds=0.1,
            default="timeout",
        )
        assert result == "timeout"

    async def test_asyncio_timeout(self) -> None:
        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await slow_operation()


class TestCancel:
    """取消测试"""

    async def test_basic_cancel(self) -> None:
        async def long_task() -> None:
            await asyncio.sleep(10.0)

        task = asyncio.create_task(long_task())
        await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()

    async def test_cancel_with_cleanup(self) -> None:
        cleaned_up = False

        async def task_with_cleanup() -> None:
            nonlocal cleaned_up
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cleaned_up = True
                raise

        task = asyncio.create_task(task_with_cleanup())
        await asyncio.sleep(0.01)

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cleaned_up


class TestShield:
    """shield 测试"""

    async def test_shield_protection(self) -> None:
        completed = False

        async def critical_operation() -> None:
            nonlocal completed
            await asyncio.sleep(0.1)
            completed = True

        inner_task = asyncio.create_task(critical_operation())

        async def main_task() -> None:
            await asyncio.shield(inner_task)

        task = asyncio.create_task(main_task())
        await asyncio.sleep(0.01)

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 等待内部任务完成
        await asyncio.sleep(0.15)
        assert completed

