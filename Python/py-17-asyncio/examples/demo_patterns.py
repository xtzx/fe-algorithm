#!/usr/bin/env python3
"""
asyncio 实战模式示例
"""

import asyncio
import random
import sys
import time

# 添加 src 到路径
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from async_lab.patterns import ConcurrentExecutor, retry_async
from async_lab.stats import LatencyStats


async def example_concurrent_executor():
    """并发执行器示例"""
    print("=" * 60)
    print("1. 并发执行器")
    print("=" * 60)

    async def process(n: int) -> int:
        delay = random.uniform(0.1, 0.3)
        await asyncio.sleep(delay)
        return n * 2

    executor = ConcurrentExecutor(max_concurrent=3, timeout=5.0)
    items = list(range(10))

    print(f"Processing {len(items)} items with max 3 concurrent...")
    start = time.perf_counter()
    results = await executor.run(process, items)
    elapsed = time.perf_counter() - start

    success_count = sum(1 for r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} in {elapsed:.2f}s")
    print(f"Results: {[r.result for r in results if r.success]}")

    print()


async def example_retry():
    """重试示例"""
    print("=" * 60)
    print("2. 带重试的异步执行")
    print("=" * 60)

    attempt_count = 0

    async def flaky_operation() -> str:
        nonlocal attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}...")

        if attempt_count < 3:
            raise ValueError(f"Failed on attempt {attempt_count}")

        return "Success!"

    result = await retry_async(
        flaky_operation,
        max_retries=3,
        retry_delay=0.1,
    )

    print(f"Result: {result}")

    print()


async def example_semaphore():
    """Semaphore 限流示例"""
    print("=" * 60)
    print("3. Semaphore 并发限制")
    print("=" * 60)

    semaphore = asyncio.Semaphore(3)
    active_count = 0

    async def limited_task(task_id: int):
        nonlocal active_count
        async with semaphore:
            active_count += 1
            print(f"  Task {task_id} started (active: {active_count})")
            await asyncio.sleep(0.2)
            active_count -= 1
            print(f"  Task {task_id} done")

    print("Running 10 tasks with max 3 concurrent...")
    await asyncio.gather(*[limited_task(i) for i in range(10)])

    print()


async def example_stats():
    """统计报表示例"""
    print("=" * 60)
    print("4. 延迟统计 (p50/p95)")
    print("=" * 60)

    stats = LatencyStats()

    async def simulate_request():
        delay = random.uniform(0.05, 0.3)
        await asyncio.sleep(delay)
        return delay

    print("Simulating 100 requests...")
    for _ in range(100):
        async with stats.timer():
            await simulate_request()

    summary = stats.summary()
    print(f"\n统计结果:")
    print(f"  总数: {summary['count']}")
    print(f"  平均: {summary['avg_ms']:.1f}ms")
    print(f"  最小: {summary['min_ms']:.1f}ms")
    print(f"  最大: {summary['max_ms']:.1f}ms")
    print(f"  P50:  {summary['p50_ms']:.1f}ms")
    print(f"  P95:  {summary['p95_ms']:.1f}ms")
    print(f"  P99:  {summary['p99_ms']:.1f}ms")

    print()


async def example_race():
    """竞争模式示例"""
    print("=" * 60)
    print("5. 竞争模式 (返回最快的结果)")
    print("=" * 60)

    async def fetch_from_source(source: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Data from {source}"

    tasks = [
        asyncio.create_task(fetch_from_source("Source A", 0.3)),
        asyncio.create_task(fetch_from_source("Source B", 0.1)),  # 最快
        asyncio.create_task(fetch_from_source("Source C", 0.2)),
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    result = list(done)[0].result()
    print(f"Winner: {result}")
    print(f"Cancelled {len(pending)} pending tasks")

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  asyncio 实战模式示例")
    print("=" * 60 + "\n")

    await example_concurrent_executor()
    await example_retry()
    await example_semaphore()
    await example_stats()
    await example_race()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

