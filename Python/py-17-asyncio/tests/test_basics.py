"""asyncio 基础测试"""

import asyncio

import pytest

from async_lab.basics import (
    AsyncCounter,
    AsyncResource,
    fetch_data,
    sleep_demo,
)


class TestBasics:
    """基础功能测试"""

    async def test_sleep_demo(self) -> None:
        result = await sleep_demo(0.1)
        assert result == "Slept for 0.1 seconds"

    async def test_fetch_data(self) -> None:
        result = await fetch_data(123, delay=0.01)
        assert result["id"] == 123
        assert result["status"] == "success"


class TestAsyncResource:
    """异步资源测试"""

    async def test_context_manager(self) -> None:
        async with AsyncResource("test") as resource:
            assert resource._connected
            result = await resource.do_work()
            assert "test" in result

        # 退出后断开连接
        assert not resource._connected

    async def test_work_without_connect(self) -> None:
        resource = AsyncResource("test")
        with pytest.raises(RuntimeError, match="Not connected"):
            await resource.do_work()


class TestAsyncCounter:
    """异步迭代器测试"""

    async def test_iteration(self) -> None:
        counter = AsyncCounter(3, delay=0.01)
        values = []

        async for value in counter:
            values.append(value)

        assert values == [0, 1, 2]

    async def test_empty_iteration(self) -> None:
        counter = AsyncCounter(0, delay=0.01)
        values = []

        async for value in counter:
            values.append(value)

        assert values == []

