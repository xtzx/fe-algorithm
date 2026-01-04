"""并发原语测试"""

import asyncio

import pytest

from async_lab.concurrency import (
    gather_with_errors,
    map_concurrent,
    run_concurrently,
)


class TestGather:
    """gather 测试"""

    async def test_run_concurrently(self) -> None:
        async def fetch(item_id: int) -> dict:
            await asyncio.sleep(0.01)
            return {"id": item_id}

        results = await run_concurrently(fetch(1), fetch(2), fetch(3))

        assert len(results) == 3
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[2]["id"] == 3

    async def test_gather_with_errors(self) -> None:
        async def good_task(n: int) -> int:
            return n * 2

        async def bad_task() -> None:
            raise ValueError("Test error")

        results = await gather_with_errors(
            good_task(1),
            bad_task(),
            good_task(3),
        )

        assert results[0] == 2
        assert isinstance(results[1], ValueError)
        assert results[2] == 6


class TestMapConcurrent:
    """并发 map 测试"""

    async def test_unlimited(self) -> None:
        async def double(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        results = await map_concurrent(double, [1, 2, 3, 4, 5])

        assert results == [2, 4, 6, 8, 10]

    async def test_limited(self) -> None:
        active_count = 0
        max_active = 0

        async def track_concurrent(n: int) -> int:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1
            return n

        await map_concurrent(track_concurrent, list(range(10)), max_concurrent=3)

        assert max_active <= 3


class TestTaskGroup:
    """TaskGroup 测试"""

    async def test_basic(self) -> None:
        results = []

        async def task(n: int) -> None:
            await asyncio.sleep(0.01)
            results.append(n)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(task(1))
            tg.create_task(task(2))
            tg.create_task(task(3))

        assert sorted(results) == [1, 2, 3]

    async def test_exception_handling(self) -> None:
        async def bad_task() -> None:
            raise ValueError("Test error")

        with pytest.raises(ExceptionGroup):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(bad_task())

