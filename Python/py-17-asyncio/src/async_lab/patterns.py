"""
实战模式

包含:
- 并发请求
- 生产者/消费者
- 限制并发数
- 工作池
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# 1. 并发请求执行器
# =============================================================================


@dataclass
class TaskResult(Generic[T]):
    """任务结果"""

    success: bool
    result: T | None = None
    error: Exception | None = None
    elapsed: float = 0.0


class ConcurrentExecutor(Generic[T, R]):
    """
    并发执行器

    支持:
    - 限制并发数
    - 超时控制
    - 错误处理
    - 结果收集

    Example:
        ```python
        async def fetch(url: str) -> dict:
            ...

        executor = ConcurrentExecutor(
            max_concurrent=10,
            timeout=30.0,
        )

        results = await executor.run(fetch, urls)
        ```
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: float | None = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore: asyncio.Semaphore | None = None

    async def run(
        self,
        func: Callable[[T], Awaitable[R]],
        items: list[T],
    ) -> list[TaskResult[R]]:
        """执行所有任务"""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._execute_one(func, item) for item in items]
        return await asyncio.gather(*tasks)

    async def _execute_one(
        self,
        func: Callable[[T], Awaitable[R]],
        item: T,
    ) -> TaskResult[R]:
        """执行单个任务"""
        assert self._semaphore is not None

        async with self._semaphore:
            start = time.perf_counter()
            try:
                if self.timeout:
                    async with asyncio.timeout(self.timeout):
                        result = await func(item)
                else:
                    result = await func(item)

                return TaskResult(
                    success=True,
                    result=result,
                    elapsed=time.perf_counter() - start,
                )
            except Exception as e:
                return TaskResult(
                    success=False,
                    error=e,
                    elapsed=time.perf_counter() - start,
                )


# =============================================================================
# 2. 生产者/消费者模式
# =============================================================================


@dataclass
class WorkItem(Generic[T]):
    """工作项"""

    data: T
    created_at: float = field(default_factory=time.time)


class ProducerConsumer(Generic[T, R]):
    """
    生产者/消费者模式

    Example:
        ```python
        async def process(item: str) -> dict:
            ...

        pc = ProducerConsumer(
            processor=process,
            num_workers=5,
            queue_size=100,
        )

        await pc.start()

        # 添加工作
        await pc.put("item1")
        await pc.put("item2")

        # 等待完成
        await pc.join()
        await pc.stop()

        results = pc.results
        ```
    """

    def __init__(
        self,
        processor: Callable[[T], Awaitable[R]],
        num_workers: int = 5,
        queue_size: int = 100,
    ) -> None:
        self.processor = processor
        self.num_workers = num_workers
        self._queue: asyncio.Queue[WorkItem[T] | None] = asyncio.Queue(queue_size)
        self._workers: list[asyncio.Task] = []
        self._results: list[TaskResult[R]] = []
        self._running = False

    @property
    def results(self) -> list[TaskResult[R]]:
        return self._results

    async def start(self) -> None:
        """启动工作者"""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self.num_workers)
        ]

    async def stop(self) -> None:
        """停止所有工作者"""
        self._running = False

        # 发送停止信号
        for _ in range(self.num_workers):
            await self._queue.put(None)

        # 等待工作者结束
        await asyncio.gather(*self._workers)
        self._workers.clear()

    async def put(self, item: T) -> None:
        """添加工作项"""
        await self._queue.put(WorkItem(data=item))

    async def join(self) -> None:
        """等待所有工作完成"""
        await self._queue.join()

    async def _worker(self, worker_id: int) -> None:
        """工作者协程"""
        while self._running:
            work_item = await self._queue.get()

            if work_item is None:
                self._queue.task_done()
                break

            start = time.perf_counter()
            try:
                result = await self.processor(work_item.data)
                self._results.append(
                    TaskResult(
                        success=True,
                        result=result,
                        elapsed=time.perf_counter() - start,
                    )
                )
            except Exception as e:
                self._results.append(
                    TaskResult(
                        success=False,
                        error=e,
                        elapsed=time.perf_counter() - start,
                    )
                )
            finally:
                self._queue.task_done()


# =============================================================================
# 3. 工作池
# =============================================================================


class WorkerPool:
    """
    工作池

    复用固定数量的工作者执行任务

    Example:
        ```python
        async def process(data: str) -> dict:
            ...

        async with WorkerPool(num_workers=10) as pool:
            results = await pool.map(process, items)
        ```
    """

    def __init__(self, num_workers: int = 10) -> None:
        self.num_workers = num_workers
        self._queue: asyncio.Queue[tuple[Any, asyncio.Future]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def __aenter__(self) -> "WorkerPool":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    async def start(self) -> None:
        """启动工作池"""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(self.num_workers)
        ]

    async def stop(self) -> None:
        """停止工作池"""
        self._running = False

        # 取消所有工作者
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(
        self,
        func: Callable[[T], Awaitable[R]],
        arg: T,
    ) -> R:
        """提交单个任务"""
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        await self._queue.put((func, arg, future))
        return await future

    async def map(
        self,
        func: Callable[[T], Awaitable[R]],
        items: list[T],
    ) -> list[R]:
        """并发执行 map"""
        futures = []
        for item in items:
            future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
            await self._queue.put((func, item, future))
            futures.append(future)

        return await asyncio.gather(*futures)

    async def _worker(self) -> None:
        """工作者协程"""
        while self._running:
            try:
                func, arg, future = await self._queue.get()
                try:
                    result = await func(arg)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            except asyncio.CancelledError:
                break


# =============================================================================
# 4. 批量处理
# =============================================================================


class BatchProcessor(Generic[T, R]):
    """
    批量处理器

    收集多个请求，批量处理

    Example:
        ```python
        async def batch_query(ids: list[int]) -> list[dict]:
            ...

        processor = BatchProcessor(
            batch_handler=batch_query,
            max_batch_size=100,
            max_wait_time=0.1,
        )

        # 单个请求会被自动批量化
        result = await processor.process(1)
        result = await processor.process(2)
        ```
    """

    def __init__(
        self,
        batch_handler: Callable[[list[T]], Awaitable[list[R]]],
        max_batch_size: int = 100,
        max_wait_time: float = 0.1,
    ) -> None:
        self.batch_handler = batch_handler
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        self._pending: list[tuple[T, asyncio.Future[R]]] = []
        self._lock = asyncio.Lock()
        self._batch_task: asyncio.Task | None = None

    async def process(self, item: T) -> R:
        """处理单个项目"""
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()

        async with self._lock:
            self._pending.append((item, future))

            # 如果达到批量大小，立即处理
            if len(self._pending) >= self.max_batch_size:
                await self._flush()
            elif self._batch_task is None:
                # 启动定时器
                self._batch_task = asyncio.create_task(self._timer())

        return await future

    async def _timer(self) -> None:
        """定时刷新"""
        await asyncio.sleep(self.max_wait_time)
        async with self._lock:
            await self._flush()
            self._batch_task = None

    async def _flush(self) -> None:
        """刷新批量"""
        if not self._pending:
            return

        batch = self._pending[:]
        self._pending.clear()

        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        try:
            results = await self.batch_handler(items)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)


# =============================================================================
# 5. 重试模式
# =============================================================================


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    retry_delay: float = 0.5,
    exponential_backoff: bool = True,
    exceptions: tuple = (Exception,),
) -> T:
    """
    带重试的异步执行

    Example:
        ```python
        result = await retry_async(
            lambda: fetch_data(url),
            max_retries=3,
            retry_delay=0.5,
        )
        ```
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_delay
                if exponential_backoff:
                    delay = retry_delay * (2**attempt)
                await asyncio.sleep(delay)

    raise last_error  # type: ignore


# =============================================================================
# 6. 扇出/扇入模式
# =============================================================================


async def fan_out_fan_in(
    items: list[T],
    processor: Callable[[T], Awaitable[R]],
    aggregator: Callable[[list[R]], Awaitable[Any]],
    max_concurrent: int = 10,
) -> Any:
    """
    扇出/扇入模式

    1. 扇出：并发处理多个项目
    2. 扇入：聚合所有结果

    Example:
        ```python
        async def fetch(url: str) -> dict:
            ...

        async def aggregate(results: list[dict]) -> dict:
            return {"total": len(results)}

        result = await fan_out_fan_in(urls, fetch, aggregate)
        ```
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_process(item: T) -> R:
        async with semaphore:
            return await processor(item)

    # 扇出
    results = await asyncio.gather(*[limited_process(item) for item in items])

    # 扇入
    return await aggregator(results)

