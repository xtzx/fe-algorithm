"""实战模式测试"""

import asyncio

import pytest

from async_lab.patterns import ConcurrentExecutor, ProducerConsumer, retry_async
from async_lab.stats import LatencyStats


class TestConcurrentExecutor:
    """并发执行器测试"""

    async def test_basic_execution(self) -> None:
        async def process(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        executor = ConcurrentExecutor(max_concurrent=5)
        results = await executor.run(process, [1, 2, 3, 4, 5])

        assert len(results) == 5
        assert all(r.success for r in results)
        assert [r.result for r in results] == [2, 4, 6, 8, 10]

    async def test_with_errors(self) -> None:
        async def process(n: int) -> int:
            if n == 3:
                raise ValueError("Test error")
            return n * 2

        executor = ConcurrentExecutor(max_concurrent=5)
        results = await executor.run(process, [1, 2, 3, 4, 5])

        assert results[0].success
        assert results[1].success
        assert not results[2].success
        assert isinstance(results[2].error, ValueError)
        assert results[3].success
        assert results[4].success


class TestProducerConsumer:
    """生产者/消费者测试"""

    async def test_basic(self) -> None:
        processed = []

        async def processor(item: int) -> int:
            await asyncio.sleep(0.01)
            result = item * 2
            processed.append(result)
            return result

        pc = ProducerConsumer(processor, num_workers=2)
        await pc.start()

        for i in range(5):
            await pc.put(i)

        await pc.join()
        await pc.stop()

        assert sorted(processed) == [0, 2, 4, 6, 8]


class TestRetryAsync:
    """异步重试测试"""

    async def test_success_first_try(self) -> None:
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await retry_async(operation, max_retries=3)

        assert result == "done"
        assert call_count == 1

    async def test_success_after_retry(self) -> None:
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "done"

        result = await retry_async(
            operation,
            max_retries=3,
            retry_delay=0.01,
        )

        assert result == "done"
        assert call_count == 3

    async def test_all_retries_failed(self) -> None:
        async def operation() -> str:
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await retry_async(operation, max_retries=2, retry_delay=0.01)


class TestLatencyStats:
    """延迟统计测试"""

    def test_basic_stats(self) -> None:
        stats = LatencyStats()

        for i in range(1, 11):
            stats.record(i * 0.1)

        assert stats.count == 10
        assert stats.min == 0.1
        assert stats.max == 1.0
        assert abs(stats.avg - 0.55) < 0.01

    def test_percentiles(self) -> None:
        stats = LatencyStats()

        for i in range(1, 101):
            stats.record(i)

        assert stats.p50 == 50
        assert stats.p90 == 90
        assert stats.p95 == 95
        assert stats.p99 == 99

    def test_timer_context(self) -> None:
        stats = LatencyStats()

        with stats.timer():
            import time

            time.sleep(0.05)

        assert stats.count == 1
        assert stats.min >= 0.05

    def test_summary(self) -> None:
        stats = LatencyStats()
        stats.record(0.1)
        stats.record(0.2)

        summary = stats.summary()

        assert "count" in summary
        assert "avg_ms" in summary
        assert "p50_ms" in summary
        assert "p95_ms" in summary

