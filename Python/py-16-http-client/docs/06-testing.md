# 测试 HTTP 客户端

> 使用 Mock 测试不同场景

## 1. 测试策略

| 层级 | 工具 | 用途 |
|------|------|------|
| 单元测试 | MockTransport | 隔离测试客户端逻辑 |
| 集成测试 | httpx 真实请求 | 测试实际 API |
| Mock 服务器 | respx | 模拟外部 API |

## 2. 使用 MockTransport

### 基础用法

```python
from http_kit import HttpClient
from http_kit.testing import MockTransport, MockResponse

def test_get_users():
    # 设置 Mock 响应
    transport = MockTransport([
        MockResponse(
            url="/users",
            json_data=[{"id": 1, "name": "Alice"}],
        ),
    ])

    # 创建客户端
    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
    )

    # 发送请求
    response = client.get("/users")

    # 验证
    assert response.status_code == 200
    assert response.json() == [{"id": 1, "name": "Alice"}]


def test_create_user():
    transport = MockTransport([
        MockResponse(
            url="/users",
            method="POST",
            status_code=201,
            json_data={"id": 2, "name": "Bob"},
        ),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
    )

    response = client.post("/users", json={"name": "Bob"})

    assert response.status_code == 201
    assert response.json()["id"] == 2
```

### 验证请求

```python
def test_request_verification():
    transport = MockTransport([
        MockResponse(url="/users", json_data=[]),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
    )

    client.get("/users", params={"page": 1})

    # 验证请求
    transport.assert_called()
    transport.assert_called_once()
    transport.assert_called_with(method="GET", url="/users")

    # 检查请求详情
    request = transport.requests[0]
    assert request.method == "GET"
    assert "page=1" in str(request.url)
```

## 3. 使用 respx

更强大的 Mock 工具：

```bash
pip install respx
```

```python
import pytest
import respx
import httpx

@respx.mock
def test_with_respx():
    # 设置 Mock
    respx.get("https://api.example.com/users").mock(
        return_value=httpx.Response(
            200,
            json=[{"id": 1, "name": "Alice"}],
        )
    )

    # 发送请求
    response = httpx.get("https://api.example.com/users")

    assert response.json() == [{"id": 1, "name": "Alice"}]
```

### 使用 pytest fixture

```python
import pytest
import respx

@pytest.fixture
def mock_api():
    with respx.mock:
        # 设置通用 Mock
        respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(200, json=[])
        )
        yield


def test_empty_users(mock_api):
    response = httpx.get("https://api.example.com/users")
    assert response.json() == []
```

## 4. 测试错误场景

### 测试超时

```python
import httpx
from http_kit import HttpClient
from http_kit.testing import MockResponse

def test_timeout():
    def slow_response(request):
        import time
        time.sleep(2)  # 模拟慢响应
        return httpx.Response(200)

    transport = MockTransport([
        MockResponse(callback=slow_response),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        timeout=0.5,
    )

    with pytest.raises(httpx.ReadTimeout):
        client.get("/slow")
```

### 测试连接错误

```python
import respx
import httpx

@respx.mock
def test_connection_error():
    respx.get("https://api.example.com/").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(httpx.ConnectError):
        httpx.get("https://api.example.com/")
```

### 测试重试

```python
from http_kit import HttpClient
from http_kit.retry import RetryConfig
from http_kit.testing import MockTransport, MockResponse

def test_retry_on_500():
    # 第一次失败，第二次成功
    responses = [
        MockResponse(status_code=500),
        MockResponse(status_code=200, json_data={"ok": True}),
    ]

    transport = MockTransport(responses)

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        retry_config=RetryConfig(max_retries=3),
    )

    response = client.get("/unstable")

    assert response.status_code == 200
    assert len(transport.requests) == 2  # 验证重试次数
```

## 5. 测试异步客户端

```python
import pytest
from http_kit import AsyncHttpClient
from http_kit.testing import AsyncMockTransport, MockResponse

@pytest.mark.asyncio
async def test_async_get():
    transport = AsyncMockTransport([
        MockResponse(url="/users", json_data=[{"id": 1}]),
    ])

    async with AsyncHttpClient(
        base_url="https://api.example.com",
        transport=transport,
    ) as client:
        response = await client.get("/users")
        assert response.json() == [{"id": 1}]
```

## 6. 测试限流

```python
import time
from http_kit import HttpClient
from http_kit.rate_limit import RateLimiter
from http_kit.testing import MockTransport, MockResponse

def test_rate_limiting():
    transport = MockTransport([
        MockResponse(status_code=200) for _ in range(10)
    ])

    limiter = RateLimiter(requests_per_second=5)

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        rate_limiter=limiter,
    )

    start = time.time()

    # 发送 10 个请求
    for _ in range(10):
        client.get("/")

    elapsed = time.time() - start

    # 5 RPS，10 个请求应该至少需要 ~2 秒
    assert elapsed >= 1.5
```

## 7. 测试 Trace ID

```python
from http_kit import HttpClient
from http_kit.tracing import TracingMiddleware, set_trace_id
from http_kit.testing import MockTransport, MockResponse

def test_trace_id_propagation():
    transport = MockTransport([
        MockResponse(status_code=200),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        middlewares=[TracingMiddleware()],
    )

    set_trace_id("test-trace-123")
    client.get("/users")

    # 验证请求包含 trace_id
    request = transport.requests[0]
    assert request.headers["X-Trace-Id"] == "test-trace-123"
```

## 8. 测试组织

```
tests/
├── conftest.py          # 共享 fixture
├── test_client.py       # 客户端测试
├── test_retry.py        # 重试测试
├── test_rate_limit.py   # 限流测试
├── test_tracing.py      # 追踪测试
└── integration/
    └── test_real_api.py # 集成测试（可选）
```

### conftest.py

```python
import pytest
from http_kit.testing import MockTransport, MockResponse

@pytest.fixture
def mock_transport():
    return MockTransport()

@pytest.fixture
def success_response():
    return MockResponse(status_code=200, json_data={"ok": True})

@pytest.fixture
def error_response():
    return MockResponse(status_code=500, text="Internal Server Error")
```

## 9. 与 JS 对比

```python
# Python with http_kit
from http_kit.testing import MockTransport

transport = MockTransport([...])
client = HttpClient(transport=transport)
```

```javascript
// JavaScript with msw
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.get('/users', (req, res, ctx) => {
    return res(ctx.json([{ id: 1 }]));
  }),
);

beforeAll(() => server.listen());
afterAll(() => server.close());
```

## 小结

| 工具 | 用途 |
|------|------|
| MockTransport | 单元测试，隔离测试 |
| respx | 更强大的 Mock |
| pytest-asyncio | 异步测试 |
| conftest.py | 共享 fixture |

