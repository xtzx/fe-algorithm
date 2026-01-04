# P16: HTTP 客户端工程化

> 构建可复用的 HTTP 客户端，处理生产环境复杂场景

## 🎯 学完后能做

- 使用 httpx 进行同步/异步 HTTP 请求
- 实现重试、限流、代理
- 构建可测试的 HTTP 客户端

## 🚀 快速开始

```bash
# 进入项目目录
cd py-16-http-client

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行示例
python examples/basic_usage.py
```

## 📁 目录结构

```
py-16-http-client/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-httpx-basics.md       # httpx 基础
│   ├── 02-advanced-config.md    # 高级配置
│   ├── 03-retry.md              # 重试策略
│   ├── 04-rate-limit.md         # 限流
│   ├── 05-observability.md      # 可观测性
│   ├── 06-testing.md            # 测试
│   ├── 07-exercises.md          # 练习题
│   └── 08-interview.md          # 面试题
├── src/http_kit/
│   ├── __init__.py
│   ├── client.py                # HTTP 客户端
│   ├── retry.py                 # 重试策略
│   ├── rate_limit.py            # 限流
│   ├── tracing.py               # 追踪
│   └── testing.py               # 测试工具
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_client.py
│   ├── test_retry.py
│   ├── test_rate_limit.py
│   └── test_tracing.py
├── examples/
│   ├── basic_usage.py
│   ├── async_usage.py
│   └── advanced_features.py
└── scripts/
    └── run_examples.sh
```

## 🆚 Python vs JavaScript 对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| HTTP 客户端 | `httpx` | `fetch` / `axios` |
| 异步请求 | `async/await` + `httpx.AsyncClient` | `async/await` + `fetch` |
| 重试 | 自定义装饰器 / `tenacity` | `axios-retry` |
| 限流 | `asyncio.Semaphore` | `p-limit` / `bottleneck` |
| Mock | `respx` / `pytest-httpx` | `msw` / `nock` |

## 🔧 核心功能

### 1. 同步/异步客户端

```python
from http_kit import HttpClient, AsyncHttpClient

# 同步
client = HttpClient(base_url="https://api.example.com")
response = client.get("/users")

# 异步
async with AsyncHttpClient(base_url="https://api.example.com") as client:
    response = await client.get("/users")
```

### 2. 重试策略

```python
from http_kit import HttpClient
from http_kit.retry import RetryConfig

client = HttpClient(
    base_url="https://api.example.com",
    retry_config=RetryConfig(
        max_retries=3,
        backoff_factor=0.5,
        retry_on_status=[500, 502, 503, 504],
    ),
)
```

### 3. 限流

```python
from http_kit import HttpClient
from http_kit.rate_limit import RateLimiter

limiter = RateLimiter(requests_per_second=10)
client = HttpClient(
    base_url="https://api.example.com",
    rate_limiter=limiter,
)
```

### 4. 可观测性

```python
from http_kit import HttpClient
from http_kit.tracing import TracingMiddleware

client = HttpClient(
    base_url="https://api.example.com",
    middlewares=[TracingMiddleware()],
)

# 自动生成 trace_id，记录请求日志
```

### 5. 测试

```python
import pytest
from http_kit.testing import MockTransport

def test_get_users():
    transport = MockTransport([
        {"url": "/users", "json": [{"id": 1, "name": "Alice"}]}
    ])
    client = HttpClient(base_url="https://api.example.com", transport=transport)

    response = client.get("/users")
    assert response.json() == [{"id": 1, "name": "Alice"}]
```

## 📚 学习路径

1. **httpx 基础** - 掌握基本请求方法
2. **高级配置** - 超时、连接池、代理
3. **重试策略** - 指数退避、错误处理
4. **限流** - 速率限制、并发控制
5. **可观测性** - 日志、追踪
6. **测试** - Mock 和集成测试

## ✅ 功能清单

- [x] 同步 HTTP 客户端
- [x] 异步 HTTP 客户端
- [x] GET/POST/PUT/DELETE/PATCH
- [x] 请求参数、头部、body
- [x] 超时配置
- [x] 连接池
- [x] 代理设置
- [x] 指数退避重试
- [x] 速率限制
- [x] 并发控制
- [x] 429 处理
- [x] 请求日志
- [x] trace_id 传递
- [x] 计时统计
- [x] Mock 测试支持

