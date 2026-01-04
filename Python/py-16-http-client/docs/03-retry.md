# 重试策略

> 处理瞬态错误，提高请求可靠性

## 1. 为什么需要重试

网络请求可能因多种原因失败：
- 网络抖动
- 服务器过载
- 临时服务不可用
- 连接超时

合理的重试策略可以提高可靠性。

## 2. 可重试的错误类型

### HTTP 状态码

| 状态码 | 含义 | 是否重试 |
|--------|------|----------|
| 408 | 请求超时 | ✓ |
| 429 | 请求过多 | ✓ (需等待) |
| 500 | 服务器内部错误 | ✓ |
| 502 | 网关错误 | ✓ |
| 503 | 服务不可用 | ✓ |
| 504 | 网关超时 | ✓ |
| 400 | 客户端错误 | ✗ |
| 401 | 未授权 | ✗ |
| 404 | 未找到 | ✗ |

### 异常类型

```python
import httpx

RETRYABLE_EXCEPTIONS = [
    httpx.ConnectError,     # 连接失败
    httpx.ReadTimeout,      # 读取超时
    httpx.WriteTimeout,     # 写入超时
    httpx.ConnectTimeout,   # 连接超时
]
```

## 3. 指数退避

避免对服务器造成额外压力：

```
重试次数  等待时间
1         0.5s
2         1.0s
3         2.0s
4         4.0s
5         8.0s
```

### 公式

```
wait_time = backoff_factor * (2 ^ retry_count)
```

### 带抖动的退避

添加随机性避免"惊群效应"：

```python
import random

def get_wait_time(retry_count: int, backoff_factor: float = 0.5) -> float:
    base_wait = backoff_factor * (2 ** retry_count)
    # 添加 0-25% 的随机抖动
    jitter = base_wait * random.random() * 0.25
    return base_wait + jitter
```

## 4. 使用 http_kit 重试

### 基础配置

```python
from http_kit import HttpClient
from http_kit.retry import RetryConfig

client = HttpClient(
    base_url="https://api.example.com",
    retry_config=RetryConfig(
        max_retries=3,
        backoff_factor=0.5,
    ),
)

# 自动重试失败的请求
response = client.get("/unstable-endpoint")
```

### 自定义配置

```python
from http_kit.retry import RetryConfig
import httpx

config = RetryConfig(
    max_retries=5,
    backoff_factor=1.0,
    max_backoff=30.0,
    jitter=True,
    retry_on_status=[429, 500, 502, 503, 504],
    retry_on_exceptions=[
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
    ],
)

client = HttpClient(retry_config=config)
```

## 5. 手动实现重试

```python
import time
import httpx

def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> httpx.Response:
    """带重试的请求"""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = client.request(method, url, **kwargs)

            # 检查是否需要重试
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    time.sleep(wait)
                    continue

            return response

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exception = e
            if attempt < max_retries:
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise

    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected error")
```

## 6. 429 特殊处理

429 Too Many Requests 需要特殊处理：

```python
import time
import httpx

def handle_rate_limit(response: httpx.Response) -> float:
    """解析 429 响应中的等待时间"""
    # 尝试 Retry-After 头
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    # 尝试 X-RateLimit-Reset 头
    reset_time = response.headers.get("X-RateLimit-Reset")
    if reset_time:
        try:
            return float(reset_time) - time.time()
        except ValueError:
            pass

    # 默认等待 60 秒
    return 60.0


def request_with_rate_limit(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    """处理 429 的请求"""
    while True:
        response = client.request(method, url, **kwargs)

        if response.status_code == 429:
            wait_time = handle_rate_limit(response)
            print(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            continue

        return response
```

## 7. 幂等性考虑

### 安全重试的方法

| HTTP 方法 | 幂等性 | 安全重试 |
|-----------|--------|----------|
| GET | ✓ | ✓ |
| HEAD | ✓ | ✓ |
| PUT | ✓ | ✓ |
| DELETE | ✓ | ✓ |
| POST | ✗ | ⚠️ 需要幂等键 |
| PATCH | ✗ | ⚠️ 需要幂等键 |

### 使用幂等键

```python
import uuid

def create_order(client: httpx.Client, order_data: dict) -> dict:
    """使用幂等键创建订单"""
    idempotency_key = str(uuid.uuid4())

    response = client.post(
        "/orders",
        json=order_data,
        headers={"Idempotency-Key": idempotency_key},
    )

    return response.json()
```

## 8. 与 JS 对比

```python
# Python
from http_kit import HttpClient
from http_kit.retry import RetryConfig

client = HttpClient(
    retry_config=RetryConfig(max_retries=3),
)
```

```javascript
// JavaScript with axios-retry
import axios from 'axios';
import axiosRetry from 'axios-retry';

const client = axios.create();
axiosRetry(client, { retries: 3 });
```

## 小结

| 概念 | 要点 |
|------|------|
| 可重试状态码 | 429, 5xx |
| 可重试异常 | 连接错误、超时 |
| 指数退避 | wait = factor * 2^n |
| 抖动 | 避免惊群效应 |
| 429 处理 | 解析 Retry-After |
| 幂等性 | POST 需要幂等键 |

