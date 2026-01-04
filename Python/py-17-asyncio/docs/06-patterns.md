# 实战模式

> 并发请求、生产者/消费者、限制并发数、统计报表

## 1. 并发请求模式

### 基础并发

```python
async def fetch_all(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
    return responses
```

### 带限流的并发

```python
async def fetch_all_limited(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(client, url):
        async with semaphore:
            return await client.get(url)

    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 带重试的请求

```python
async def fetch_with_retry(client, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
```

## 2. 生产者/消费者模式

### 基础实现

```python
async def producer(queue: asyncio.Queue):
    for item in items:
        await queue.put(item)
    # 发送结束信号
    for _ in range(NUM_CONSUMERS):
        await queue.put(None)

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=100)

    await asyncio.gather(
        producer(queue),
        consumer(queue),
        consumer(queue),
    )
```

### 多生产者/多消费者

```python
class WorkerPool:
    def __init__(self, num_workers=5):
        self.queue = asyncio.Queue()
        self.num_workers = num_workers
        self.workers = []

    async def start(self):
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.num_workers)
        ]

    async def stop(self):
        for _ in range(self.num_workers):
            await self.queue.put(None)
        await asyncio.gather(*self.workers)

    async def submit(self, item):
        await self.queue.put(item)

    async def _worker(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            await self.process(item)
            self.queue.task_done()
```

## 3. 限制并发数

### Semaphore 方式

```python
semaphore = asyncio.Semaphore(10)

async def limited_operation():
    async with semaphore:
        return await operation()
```

### 封装为执行器

```python
class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(self, coro):
        async with self._semaphore:
            return await coro

# 使用
limiter = ConcurrencyLimiter(10)
results = await asyncio.gather(*[
    limiter.run(process(item)) for item in items
])
```

## 4. 批量处理模式

```python
class BatchProcessor:
    def __init__(self, handler, batch_size=100, max_wait=0.1):
        self.handler = handler
        self.batch_size = batch_size
        self.max_wait = max_wait
        self._pending = []
        self._lock = asyncio.Lock()

    async def process(self, item):
        future = asyncio.get_running_loop().create_future()

        async with self._lock:
            self._pending.append((item, future))

            if len(self._pending) >= self.batch_size:
                await self._flush()

        return await future

    async def _flush(self):
        if not self._pending:
            return

        items = [item for item, _ in self._pending]
        futures = [future for _, future in self._pending]
        self._pending.clear()

        results = await self.handler(items)

        for future, result in zip(futures, results):
            future.set_result(result)
```

## 5. 扇出/扇入模式

```python
async def fan_out_fan_in(items, processor, aggregator):
    # 扇出：并发处理
    results = await asyncio.gather(*[
        processor(item) for item in items
    ])

    # 扇入：聚合结果
    return await aggregator(results)

# 使用
async def fetch(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)

async def merge(responses):
    return [r.json() for r in responses]

data = await fan_out_fan_in(urls, fetch, merge)
```

## 6. 统计报表 (p50/p95)

```python
class LatencyStats:
    def __init__(self):
        self._latencies = []

    def record(self, latency):
        self._latencies.append(latency)

    @property
    def p50(self):
        return self._percentile(50)

    @property
    def p95(self):
        return self._percentile(95)

    @property
    def p99(self):
        return self._percentile(99)

    def _percentile(self, p):
        if not self._latencies:
            return 0
        sorted_data = sorted(self._latencies)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def summary(self):
        return {
            "count": len(self._latencies),
            "avg_ms": sum(self._latencies) / len(self._latencies) * 1000,
            "p50_ms": self.p50 * 1000,
            "p95_ms": self.p95 * 1000,
            "p99_ms": self.p99 * 1000,
        }

# 使用
stats = LatencyStats()

for url in urls:
    start = time.perf_counter()
    await fetch(url)
    stats.record(time.perf_counter() - start)

print(stats.summary())
```

## 7. 超时竞争模式

```python
async def race_with_timeout(coros, timeout=5.0):
    """返回第一个完成的结果"""
    tasks = [asyncio.create_task(coro) for coro in coros]

    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if done:
            return list(done)[0].result()
        raise TimeoutError("All tasks timed out")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

## 8. 完整示例：并发爬虫

```python
async def concurrent_crawler(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = LatencyStats()
    results = []
    errors = []

    async def fetch_one(url):
        async with semaphore:
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=10.0)
                    results.append({"url": url, "status": response.status_code})
            except Exception as e:
                errors.append({"url": url, "error": str(e)})
            finally:
                stats.record(time.perf_counter() - start)

    await asyncio.gather(*[fetch_one(url) for url in urls])

    return {
        "results": results,
        "errors": errors,
        "stats": stats.summary(),
    }
```

## 小结

| 模式 | 用途 | 关键技术 |
|------|------|----------|
| 并发请求 | 批量 HTTP | gather + Semaphore |
| 生产者/消费者 | 任务队列 | asyncio.Queue |
| 限流 | 控制并发 | Semaphore |
| 批量处理 | 请求合并 | 累积 + 刷新 |
| 扇出/扇入 | 分布式处理 | gather + 聚合 |
| 统计 | 性能监控 | 百分位数 |

