#!/usr/bin/env python3
"""
生产者/消费者模式示例
"""

import asyncio
import random
import sys
import time

# 添加 src 到路径
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from async_lab.patterns import ProducerConsumer


async def example_basic_queue():
    """基础队列示例"""
    print("=" * 60)
    print("1. 基础 asyncio.Queue")
    print("=" * 60)

    queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=5)
    results = []

    async def producer():
        for i in range(10):
            await queue.put(i)
            print(f"  Produced: {i}, queue size: {queue.qsize()}")
            await asyncio.sleep(0.05)
        await queue.put(None)  # 结束信号

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            await asyncio.sleep(0.1)  # 模拟处理
            results.append(item * 2)
            print(f"  Consumed: {item} -> {item * 2}")
            queue.task_done()

    await asyncio.gather(producer(), consumer())

    print(f"\nResults: {results}")
    print()


async def example_multi_consumer():
    """多消费者示例"""
    print("=" * 60)
    print("2. 多消费者")
    print("=" * 60)

    queue: asyncio.Queue[int | None] = asyncio.Queue()
    results: list[tuple[int, int]] = []
    num_consumers = 3

    async def producer():
        for i in range(20):
            await queue.put(i)
            await asyncio.sleep(0.02)
        # 发送结束信号
        for _ in range(num_consumers):
            await queue.put(None)

    async def consumer(consumer_id: int):
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            await asyncio.sleep(random.uniform(0.05, 0.15))
            results.append((consumer_id, item))
            queue.task_done()

    start = time.perf_counter()

    await asyncio.gather(
        producer(),
        *[consumer(i) for i in range(num_consumers)],
    )

    elapsed = time.perf_counter() - start
    print(f"Processed {len(results)} items in {elapsed:.2f}s")
    print(f"Per consumer: {dict((i, sum(1 for r in results if r[0] == i)) for i in range(num_consumers))}")

    print()


async def example_producer_consumer_class():
    """使用 ProducerConsumer 类"""
    print("=" * 60)
    print("3. ProducerConsumer 类")
    print("=" * 60)

    async def processor(item: int) -> dict:
        delay = random.uniform(0.05, 0.15)
        await asyncio.sleep(delay)
        return {"input": item, "output": item * 2, "delay": delay}

    pc: ProducerConsumer[int, dict] = ProducerConsumer(
        processor=processor,
        num_workers=3,
        queue_size=10,
    )

    await pc.start()

    # 添加工作项
    for i in range(15):
        await pc.put(i)

    # 等待所有工作完成
    await pc.join()
    await pc.stop()

    # 查看结果
    success_count = sum(1 for r in pc.results if r.success)
    print(f"Processed: {success_count}/{len(pc.results)}")

    total_delay = sum(r.result["delay"] for r in pc.results if r.success)
    print(f"Total processing time: {total_delay:.2f}s")

    print()


async def example_with_backpressure():
    """带背压的生产者/消费者"""
    print("=" * 60)
    print("4. 背压控制 (maxsize)")
    print("=" * 60)

    queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=3)
    produced_times = []
    consumed_times = []

    async def fast_producer():
        for i in range(10):
            start = time.perf_counter()
            await queue.put(i)  # 队列满时会阻塞
            produced_times.append((i, time.perf_counter() - start))
            print(f"  Produced {i} (wait: {produced_times[-1][1]:.3f}s)")

        await queue.put(None)

    async def slow_consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            await asyncio.sleep(0.2)  # 慢速消费
            consumed_times.append(item)
            queue.task_done()

    await asyncio.gather(fast_producer(), slow_consumer())

    blocked_count = sum(1 for _, wait in produced_times if wait > 0.01)
    print(f"\nProducer blocked {blocked_count} times due to backpressure")

    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  生产者/消费者模式示例")
    print("=" * 60 + "\n")

    await example_basic_queue()
    await example_multi_consumer()
    await example_producer_consumer_class()
    await example_with_backpressure()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

