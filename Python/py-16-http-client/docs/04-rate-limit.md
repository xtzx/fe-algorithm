# 限流

> 请求速率限制与并发控制

## 1. 为什么需要限流

- 遵守 API 速率限制
- 避免服务器过载
- 防止被封禁
- 保护下游服务

## 2. 令牌桶算法

最常用的限流算法：

```
           ┌─────────────┐
           │  令牌桶     │
           │  ○ ○ ○ ○    │ ← 以固定速率填充
           │  ○ ○ ○      │
           └──────┬──────┘
                  │
                  ▼
              每次请求消耗一个令牌
```

### 特点

- 平滑限流
- 支持突发请求
- 简单高效

## 3. 使用 http_kit 限流

### 基础使用

```python
from http_kit import HttpClient
from http_kit.rate_limit import RateLimiter

# 每秒最多 10 个请求
limiter = RateLimiter(requests_per_second=10)

client = HttpClient(
    base_url="https://api.example.com",
    rate_limiter=limiter,
)

# 请求自动限流
for i in range(100):
    response = client.get("/data")
```

### 突发支持

```python
# 允许突发 5 个请求
limiter = RateLimiter(
    requests_per_second=10,
    burst=5,  # 突发容量
)
```

## 4. 手动实现令牌桶

```python
import time
import threading

class TokenBucket:
    """令牌桶限流器"""

    def __init__(
        self,
        rate: float,     # 每秒填充速率
        capacity: int,   # 桶容量
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self):
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # 计算应补充的令牌
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)

    def acquire(self, timeout: float | None = None) -> bool:
        """获取令牌，阻塞直到获得或超时"""
        start = time.monotonic()

        while True:
            with self.lock:
                self._refill()

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

                # 计算需要等待的时间
                wait = (1 - self.tokens) / self.rate

            # 检查超时
            if timeout is not None:
                if time.monotonic() - start + wait > timeout:
                    return False

            time.sleep(min(wait, 0.1))
```

## 5. 并发控制

限制同时进行的请求数量：

```python
from http_kit.rate_limit import ConcurrencyLimiter

# 最多 5 个并发请求
limiter = ConcurrencyLimiter(max_concurrent=5)

def fetch_data(url: str):
    with limiter:
        return client.get(url)
```

### 异步并发控制

```python
import asyncio
from http_kit.rate_limit import AsyncConcurrencyLimiter

limiter = AsyncConcurrencyLimiter(max_concurrent=10)

async def fetch_data(url: str):
    async with limiter:
        return await client.get(url)

# 并发执行 100 个请求，但同时最多 10 个
async def main():
    urls = [f"/data/{i}" for i in range(100)]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
```

## 6. 429 处理

当收到 429 响应时：

```python
from http_kit.rate_limit import handle_429
import time

def request_with_backoff(client, url: str):
    while True:
        response = client.get(url)

        if response.status_code == 429:
            wait_time = handle_429(response)
            print(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            continue

        return response
```

## 7. 滑动窗口限流

另一种常见算法：

```python
import time
from collections import deque
import threading

class SlidingWindowLimiter:
    """滑动窗口限流器"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """尝试获取许可"""
        now = time.monotonic()

        with self.lock:
            # 移除过期的请求记录
            cutoff = now - self.window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                return False

            # 记录本次请求
            self.requests.append(now)
            return True
```

## 8. 组合使用

同时使用多种限流策略：

```python
from http_kit import HttpClient
from http_kit.rate_limit import RateLimiter, ConcurrencyLimiter

# 速率限制
rate_limiter = RateLimiter(requests_per_second=10)

# 并发限制
concurrency_limiter = ConcurrencyLimiter(max_concurrent=5)

client = HttpClient(
    base_url="https://api.example.com",
    rate_limiter=rate_limiter,
)

def fetch_with_limits(url: str):
    with concurrency_limiter:
        return client.get(url)
```

## 9. 与 JS 对比

```python
# Python
from http_kit.rate_limit import RateLimiter

limiter = RateLimiter(requests_per_second=10)
limiter.acquire()
# 发送请求
```

```javascript
// JavaScript with p-limit
import pLimit from 'p-limit';

const limit = pLimit(10);  // 并发限制
await limit(() => fetch(url));
```

```javascript
// JavaScript with bottleneck
import Bottleneck from 'bottleneck';

const limiter = new Bottleneck({
  minTime: 100,  // 100ms 间隔 = 10 RPS
});

await limiter.schedule(() => fetch(url));
```

## 10. 最佳实践

### 速率限制参考

| API | 通常限制 |
|-----|---------|
| GitHub API | 5000/小时 |
| Twitter API | 300/15分钟 |
| OpenAI API | 60/分钟 |
| 一般 REST API | 100-1000/分钟 |

### 配置建议

```python
# 留有余量，不要用满配额
actual_limit = 100  # API 限制
safe_limit = actual_limit * 0.8  # 使用 80%

limiter = RateLimiter(
    requests_per_second=safe_limit / 60,
    burst=10,
)
```

## 小结

| 概念 | 要点 |
|------|------|
| 令牌桶 | 平滑限流，支持突发 |
| 并发控制 | 限制同时请求数 |
| 滑动窗口 | 精确的时间窗口统计 |
| 429 处理 | 解析 Retry-After |
| 安全余量 | 使用 80% 配额 |

