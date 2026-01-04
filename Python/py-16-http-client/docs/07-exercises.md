# 练习题

## 练习 1：基础 GET 请求（⭐）

使用 httpx 获取 JSONPlaceholder API 的用户列表：

```python
# 目标 URL: https://jsonplaceholder.typicode.com/users
# 要求:
# 1. 使用 httpx.Client
# 2. 打印用户数量
# 3. 打印第一个用户的名字
```

---

## 练习 2：POST 创建资源（⭐）

向 JSONPlaceholder API 创建一个新的 post：

```python
# 目标 URL: https://jsonplaceholder.typicode.com/posts
# 要求:
# 1. POST JSON 数据: {"title": "foo", "body": "bar", "userId": 1}
# 2. 验证返回状态码为 201
# 3. 打印返回的 id
```

---

## 练习 3：超时处理（⭐）

实现一个带超时处理的请求函数：

```python
def fetch_with_timeout(url: str, timeout: float = 5.0) -> dict | None:
    """
    发送 GET 请求，超时返回 None

    Args:
        url: 请求 URL
        timeout: 超时时间（秒）

    Returns:
        JSON 数据或 None（超时时）
    """
    pass
```

---

## 练习 4：实现指数退避（⭐⭐）

实现一个指数退避函数：

```python
def exponential_backoff(
    attempt: int,
    base: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> float:
    """
    计算指数退避等待时间

    Args:
        attempt: 当前重试次数（从 1 开始）
        base: 基础等待时间
        max_delay: 最大等待时间
        jitter: 是否添加随机抖动

    Returns:
        等待时间（秒）
    """
    pass

# 测试
assert 0.5 <= exponential_backoff(1, jitter=False) <= 1.0
assert 1.0 <= exponential_backoff(2, jitter=False) <= 2.0
```

---

## 练习 5：带重试的请求（⭐⭐）

实现一个带重试功能的请求函数：

```python
import httpx

def request_with_retry(
    client: httpx.Client,
    url: str,
    max_retries: int = 3,
    retry_on_status: list[int] = [500, 502, 503, 504],
) -> httpx.Response:
    """
    带重试的 GET 请求

    要求:
    1. 最多重试 max_retries 次
    2. 使用指数退避
    3. 仅在 retry_on_status 状态码时重试
    """
    pass
```

---

## 练习 6：令牌桶限流器（⭐⭐）

实现一个简单的令牌桶限流器：

```python
import time
import threading

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒填充令牌数
            capacity: 桶容量
        """
        pass

    def acquire(self) -> bool:
        """
        获取一个令牌（阻塞直到获得）
        """
        pass

    def try_acquire(self) -> bool:
        """
        尝试获取令牌（非阻塞）
        """
        pass
```

---

## 练习 7：并发请求（⭐⭐）

使用异步客户端并发获取多个用户：

```python
import asyncio
import httpx

async def fetch_users(user_ids: list[int]) -> list[dict]:
    """
    并发获取多个用户信息

    Args:
        user_ids: 用户 ID 列表

    Returns:
        用户信息列表

    要求:
    1. 使用 AsyncClient
    2. 使用 asyncio.gather 并发请求
    3. 限制并发数为 5
    """
    pass

# 测试
users = asyncio.run(fetch_users([1, 2, 3, 4, 5]))
```

---

## 练习 8：trace_id 传递（⭐⭐）

实现 trace_id 的生成和传递：

```python
from contextvars import ContextVar
import uuid

# 实现以下功能:
# 1. 使用 ContextVar 存储 trace_id
# 2. 实现 get_trace_id() 和 set_trace_id()
# 3. 在请求头中传递 X-Trace-Id
```

---

## 练习 9：Mock 测试（⭐⭐⭐）

为以下客户端编写测试：

```python
class UserClient:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url)

    def get_user(self, user_id: int) -> dict:
        response = self.client.get(f"/users/{user_id}")
        response.raise_for_status()
        return response.json()

    def create_user(self, name: str, email: str) -> dict:
        response = self.client.post(
            "/users",
            json={"name": name, "email": email},
        )
        response.raise_for_status()
        return response.json()

# 要求:
# 1. 测试 get_user 成功场景
# 2. 测试 get_user 404 场景
# 3. 测试 create_user 成功场景
```

---

## 练习 10：处理 429 响应（⭐⭐⭐）

实现 429 响应处理：

```python
import httpx
import time

def handle_rate_limit(response: httpx.Response) -> float:
    """
    解析 429 响应，返回应等待的时间

    要求:
    1. 解析 Retry-After 头（秒数或日期）
    2. 解析 X-RateLimit-Reset 头（Unix 时间戳）
    3. 如果都没有，返回默认 60 秒
    """
    pass

def request_with_rate_limit_handling(
    client: httpx.Client,
    url: str,
    max_attempts: int = 3,
) -> httpx.Response:
    """
    自动处理 429 的请求
    """
    pass
```

---

## 练习 11：请求指标收集（⭐⭐⭐）

实现一个请求指标收集器：

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class RequestMetric:
    method: str
    url: str
    status_code: int
    elapsed_ms: float
    timestamp: float

class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        pass

    def record(self, metric: RequestMetric):
        """记录一个请求指标"""
        pass

    def get_summary(self) -> dict[str, Any]:
        """
        返回摘要:
        - total_requests: 总请求数
        - avg_latency_ms: 平均延迟
        - error_rate: 错误率 (4xx/5xx)
        - status_distribution: 状态码分布
        """
        pass
```

---

## 练习 12：完整 HTTP 客户端封装（⭐⭐⭐⭐）

封装一个生产级 HTTP 客户端：

```python
class ApiClient:
    """
    要求:
    1. 支持同步和异步
    2. 配置 base_url、超时、默认头
    3. 支持重试（指数退避）
    4. 支持限流
    5. 自动添加 trace_id
    6. 请求日志
    7. 错误处理
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: float = 10.0,  # RPS
    ):
        pass

    def get(self, path: str, **kwargs) -> dict:
        pass

    def post(self, path: str, json: dict, **kwargs) -> dict:
        pass

    # ... 其他方法
```

---

## 练习答案提示

1. 使用 `httpx.Client()` 的上下文管理器
2. `response.json()` 获取 JSON，检查 `status_code`
3. 捕获 `httpx.TimeoutException`
4. `min(base * 2**attempt, max_delay)` + 随机抖动
5. 循环 + try/except + `time.sleep()`
6. 使用 `threading.Lock` 保证线程安全
7. `asyncio.Semaphore` 限制并发
8. `ContextVar` 存储，协程安全
9. 使用 `MockTransport` 或 `respx`
10. 解析 `Retry-After` 头，支持秒数和 HTTP-date 格式
11. 使用列表存储，计算统计值
12. 组合使用各个模块

