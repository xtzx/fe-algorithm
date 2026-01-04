# 练习题

## 练习 1：基础异步函数（⭐）

编写一个异步函数，模拟延迟后返回结果：

```python
async def delayed_greeting(name: str, delay: float) -> str:
    """
    等待 delay 秒后，返回问候语

    Example:
        greeting = await delayed_greeting("Alice", 1.0)
        # 1秒后返回 "Hello, Alice!"
    """
    pass
```

---

## 练习 2：并发执行（⭐）

使用 `asyncio.gather` 并发执行多个任务：

```python
async def fetch_all_users(user_ids: list[int]) -> list[dict]:
    """
    并发获取多个用户信息

    使用 asyncio.gather 并发执行
    """
    pass

# 测试
users = await fetch_all_users([1, 2, 3, 4, 5])
```

---

## 练习 3：超时控制（⭐）

实现带超时的操作：

```python
async def fetch_with_timeout(url: str, timeout: float) -> dict | None:
    """
    带超时的请求

    超时返回 None
    """
    pass
```

---

## 练习 4：任务取消（⭐⭐）

实现可取消的后台任务：

```python
async def cancellable_counter():
    """
    每秒打印计数，直到被取消

    取消时打印 "Counter stopped at X"
    """
    pass

# 使用
task = asyncio.create_task(cancellable_counter())
await asyncio.sleep(5)
task.cancel()
await task  # 应该打印 "Counter stopped at 5"
```

---

## 练习 5：限制并发（⭐⭐）

实现并发限制器：

```python
async def fetch_all_limited(
    urls: list[str],
    max_concurrent: int = 5,
) -> list[dict]:
    """
    并发获取 URL，但最多同时 max_concurrent 个

    使用 Semaphore 实现
    """
    pass
```

---

## 练习 6：生产者/消费者（⭐⭐）

实现生产者/消费者模式：

```python
async def producer_consumer(
    items: list[int],
    num_consumers: int = 3,
) -> list[int]:
    """
    生产者添加 items 到队列
    消费者处理：每个 item * 2

    返回所有处理结果
    """
    pass

# 测试
results = await producer_consumer([1, 2, 3, 4, 5], num_consumers=2)
# results = [2, 4, 6, 8, 10]
```

---

## 练习 7：TaskGroup 使用（⭐⭐）

使用 TaskGroup 管理并发任务：

```python
async def process_with_taskgroup(items: list[int]) -> list[int]:
    """
    使用 TaskGroup 并发处理

    处理：item * 2
    一个失败不影响其他
    """
    pass
```

---

## 练习 8：带重试的请求（⭐⭐⭐）

实现带重试的异步请求：

```python
async def retry_request(
    url: str,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> dict:
    """
    带指数退避重试的请求

    - 最多重试 max_retries 次
    - 指数退避延迟
    """
    pass
```

---

## 练习 9：并发统计（⭐⭐⭐）

实现请求统计：

```python
class RequestStats:
    """
    统计请求延迟

    支持:
    - record(latency): 记录延迟
    - p50, p95, p99: 百分位数
    - summary(): 返回统计摘要
    """
    pass
```

---

## 练习 10：Event 同步（⭐⭐⭐）

使用 Event 实现任务同步：

```python
async def synchronized_tasks(num_tasks: int) -> list[str]:
    """
    启动 num_tasks 个任务
    所有任务准备好后，同时开始执行

    返回每个任务的完成时间
    """
    pass
```

---

## 练习 11：异步迭代器（⭐⭐⭐）

实现异步分页迭代器：

```python
class AsyncPaginator:
    """
    异步分页迭代器

    使用:
        async for page in AsyncPaginator(url, page_size=10):
            print(page)
    """

    def __init__(self, base_url: str, page_size: int = 10):
        pass

    def __aiter__(self):
        pass

    async def __anext__(self):
        pass
```

---

## 练习 12：竞争模式（⭐⭐⭐）

实现从多个源获取数据，返回最快的：

```python
async def race_fetch(urls: list[str], timeout: float = 5.0) -> dict:
    """
    并发请求多个 URL
    返回第一个成功的结果
    取消其他请求
    """
    pass
```

---

## 练习 13：结构化并发（⭐⭐⭐⭐）

实现安全的任务管理器：

```python
class TaskManager:
    """
    结构化并发任务管理器

    - start_task(): 启动任务
    - cancel_all(): 取消所有任务
    - wait(): 等待所有任务完成
    - 退出时自动清理
    """
    pass
```

---

## 练习 14：批量处理器（⭐⭐⭐⭐）

实现请求批量化：

```python
class BatchProcessor:
    """
    批量处理器

    - 收集多个请求
    - 达到 batch_size 或超时后批量处理
    - 返回各自的结果
    """

    def __init__(self, handler, batch_size=100, max_wait=0.1):
        pass

    async def process(self, item):
        """处理单个项目，自动批量化"""
        pass
```

---

## 练习 15：完整爬虫（⭐⭐⭐⭐⭐）

实现一个完整的异步爬虫：

```python
class AsyncCrawler:
    """
    异步爬虫

    功能:
    - 并发控制（max_concurrent）
    - 请求重试
    - 超时控制
    - 统计报表（p50/p95）
    - 错误收集
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        pass

    async def crawl(self, urls: list[str]) -> dict:
        """
        爬取 URLs

        返回:
        {
            "results": [...],
            "errors": [...],
            "stats": {...},
        }
        """
        pass
```

---

## 练习答案提示

1. 使用 `asyncio.sleep(delay)`
2. `await asyncio.gather(*[fetch(id) for id in ids])`
3. `async with asyncio.timeout(timeout)`
4. try/except `asyncio.CancelledError`，清理后 raise
5. `asyncio.Semaphore(max_concurrent)`
6. `asyncio.Queue()` + 多个消费者任务
7. `async with asyncio.TaskGroup() as tg`
8. 循环 + try/except + 指数退避
9. 列表存储 + 排序计算百分位数
10. `asyncio.Event()` + wait()
11. `__aiter__` 返回 self，`__anext__` 返回下一页
12. `asyncio.wait(FIRST_COMPLETED)` + 取消 pending
13. 保存任务引用 + finally 清理
14. 收集请求 + Future + 超时刷新
15. 组合以上所有技术

