"""
同步原语

包含:
- Lock
- Semaphore
- Event
- Condition
- Queue
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# 1. Lock
# =============================================================================


class SharedCounter:
    """
    使用 Lock 保护共享状态

    Example:
        ```python
        counter = SharedCounter()
        await asyncio.gather(
            counter.increment(),
            counter.increment(),
        )
        print(counter.value)  # 2
        ```
    """

    def __init__(self) -> None:
        self._value = 0
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        return self._value

    async def increment(self) -> int:
        async with self._lock:
            # 模拟一些处理时间
            await asyncio.sleep(0.01)
            self._value += 1
            return self._value

    async def decrement(self) -> int:
        async with self._lock:
            await asyncio.sleep(0.01)
            self._value -= 1
            return self._value


async def lock_demo():
    """
    Lock 使用示例
    """
    counter = SharedCounter()

    async def increment_many(n: int):
        for _ in range(n):
            await counter.increment()

    # 并发执行
    await asyncio.gather(
        increment_many(10),
        increment_many(10),
        increment_many(10),
    )

    print(f"Final value: {counter.value}")  # 应该是 30
    return counter.value


# =============================================================================
# 2. Semaphore
# =============================================================================


async def semaphore_demo():
    """
    Semaphore 使用示例：限制并发数
    """
    semaphore = asyncio.Semaphore(3)  # 最多 3 个并发
    active_count = 0

    async def limited_task(task_id: int):
        nonlocal active_count

        async with semaphore:
            active_count += 1
            current_active = active_count
            print(f"Task {task_id} started (active: {current_active})")

            await asyncio.sleep(0.2)

            active_count -= 1
            print(f"Task {task_id} finished")

            return task_id

    # 同时启动 10 个任务，但最多 3 个并发
    results = await asyncio.gather(*[limited_task(i) for i in range(10)])

    return results


class RateLimiter:
    """
    使用 Semaphore 实现速率限制

    Example:
        ```python
        limiter = RateLimiter(10)  # 10 并发
        async with limiter:
            await do_work()
        ```
    """

    def __init__(self, max_concurrent: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    async def __aenter__(self) -> "RateLimiter":
        await self._semaphore.acquire()
        self._active += 1
        return self

    async def __aexit__(self, *args) -> None:
        self._active -= 1
        self._semaphore.release()


# =============================================================================
# 3. Event
# =============================================================================


async def event_demo():
    """
    Event 使用示例：任务间同步
    """
    event = asyncio.Event()
    results = []

    async def waiter(waiter_id: int):
        print(f"Waiter {waiter_id} waiting...")
        await event.wait()
        print(f"Waiter {waiter_id} proceeding!")
        results.append(waiter_id)

    async def setter():
        print("Setter: preparing...")
        await asyncio.sleep(0.5)
        print("Setter: setting event!")
        event.set()

    # 启动等待者和设置者
    await asyncio.gather(
        waiter(1),
        waiter(2),
        waiter(3),
        setter(),
    )

    print(f"All waiters finished: {results}")
    return results


class AsyncGate:
    """
    使用 Event 实现的门控

    Example:
        ```python
        gate = AsyncGate()

        # 消费者等待门打开
        await gate.wait()

        # 生产者打开门
        gate.open()
        ```
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def open(self) -> None:
        self._event.set()

    def close(self) -> None:
        self._event.clear()

    async def wait(self) -> None:
        await self._event.wait()

    @property
    def is_open(self) -> bool:
        return self._event.is_set()


# =============================================================================
# 4. Condition
# =============================================================================


async def condition_demo():
    """
    Condition 使用示例：复杂的同步条件
    """
    condition = asyncio.Condition()
    queue: list[int] = []
    max_size = 3

    async def producer():
        for i in range(10):
            async with condition:
                # 等待队列有空间
                while len(queue) >= max_size:
                    print(f"Producer: queue full, waiting...")
                    await condition.wait()

                queue.append(i)
                print(f"Producer: added {i}, queue={queue}")
                condition.notify_all()

            await asyncio.sleep(0.1)

    async def consumer(consumer_id: int):
        consumed = []
        for _ in range(5):  # 每个消费者消费 5 个
            async with condition:
                # 等待队列有数据
                while not queue:
                    print(f"Consumer {consumer_id}: queue empty, waiting...")
                    await condition.wait()

                item = queue.pop(0)
                consumed.append(item)
                print(f"Consumer {consumer_id}: got {item}, queue={queue}")
                condition.notify_all()

            await asyncio.sleep(0.15)

        return consumed

    # 1 个生产者，2 个消费者
    results = await asyncio.gather(
        producer(),
        consumer(1),
        consumer(2),
    )

    print(f"Consumer 1 got: {results[1]}")
    print(f"Consumer 2 got: {results[2]}")

    return results[1], results[2]


# =============================================================================
# 5. Queue
# =============================================================================


async def queue_demo():
    """
    asyncio.Queue 使用示例
    """
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=5)
    produced = []
    consumed = []

    async def producer():
        for i in range(10):
            await queue.put(i)
            produced.append(i)
            print(f"Produced: {i}, queue size: {queue.qsize()}")
            await asyncio.sleep(0.05)

        # 发送结束信号
        await queue.put(-1)

    async def consumer():
        while True:
            item = await queue.get()
            if item == -1:
                break
            consumed.append(item)
            print(f"Consumed: {item}, queue size: {queue.qsize()}")
            await asyncio.sleep(0.1)
            queue.task_done()

    await asyncio.gather(producer(), consumer())

    print(f"Produced: {produced}")
    print(f"Consumed: {consumed}")

    return consumed


class AsyncQueue:
    """
    增强的异步队列

    支持:
    - 超时
    - 优先级（可选）
    - 关闭
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
        self._closed = False

    async def put(self, item: T, timeout: float | None = None) -> bool:
        """放入元素"""
        if self._closed:
            raise RuntimeError("Queue is closed")

        if timeout is None:
            await self._queue.put(item)
            return True
        else:
            try:
                await asyncio.wait_for(self._queue.put(item), timeout)
                return True
            except TimeoutError:
                return False

    async def get(self, timeout: float | None = None) -> T | None:
        """获取元素"""
        if timeout is None:
            return await self._queue.get()
        else:
            try:
                return await asyncio.wait_for(self._queue.get(), timeout)
            except TimeoutError:
                return None

    def close(self) -> None:
        """关闭队列"""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()


# =============================================================================
# 6. Barrier (Python 3.11+)
# =============================================================================


async def barrier_demo():
    """
    Barrier 使用示例：等待所有任务到达同步点
    """
    barrier = asyncio.Barrier(3)  # 3 个参与者
    results = []

    async def worker(worker_id: int):
        print(f"Worker {worker_id}: phase 1")
        await asyncio.sleep(worker_id * 0.1)  # 不同的工作时间

        print(f"Worker {worker_id}: waiting at barrier")
        await barrier.wait()  # 等待所有人

        print(f"Worker {worker_id}: phase 2")
        results.append(worker_id)

        return worker_id

    await asyncio.gather(
        worker(0),
        worker(1),
        worker(2),
    )

    print(f"All workers completed: {results}")
    return results

