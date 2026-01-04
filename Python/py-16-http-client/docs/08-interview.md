# 面试题

## 1. httpx 和 requests 的区别？

**答案**：

| 特性 | httpx | requests |
|------|-------|----------|
| 异步支持 | ✓ 原生 async/await | ✗ 不支持 |
| HTTP/2 | ✓ | ✗ |
| 类型注解 | ✓ 完整 | 部分 |
| 维护状态 | 活跃开发 | 维护模式 |
| API 风格 | 兼容 requests | - |
| 依赖 | 较少 | urllib3 |

**选择建议**：
- 新项目推荐 httpx
- 需要异步必选 httpx
- 存量项目可继续用 requests

---

## 2. 如何实现请求重试？

**答案**：

**关键要素**：
1. 确定可重试的条件（状态码、异常类型）
2. 使用指数退避避免压垮服务器
3. 设置最大重试次数

**实现示例**：

```python
import time
import httpx

def request_with_retry(client, url, max_retries=3):
    retry_on_status = [429, 500, 502, 503, 504]

    for attempt in range(max_retries + 1):
        try:
            response = client.get(url)

            if response.status_code in retry_on_status:
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)  # 指数退避
                    time.sleep(wait)
                    continue

            return response

        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < max_retries:
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise
```

**最佳实践**：
- 只重试幂等请求（GET、PUT、DELETE）
- POST 需要幂等键
- 添加抖动避免惊群效应

---

## 3. 如何处理 429 Too Many Requests？

**答案**：

**处理步骤**：
1. 解析 `Retry-After` 头获取等待时间
2. 解析 `X-RateLimit-Reset` 获取重置时间
3. 等待后重试

```python
def handle_429(response):
    # 1. 尝试 Retry-After
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        return float(retry_after)

    # 2. 尝试 X-RateLimit-Reset
    reset = response.headers.get("X-RateLimit-Reset")
    if reset:
        return max(0, float(reset) - time.time())

    # 3. 默认等待
    return 60.0
```

**预防措施**：
- 客户端限流（令牌桶）
- 遵守 API 速率限制
- 留有余量（使用 80% 配额）

---

## 4. 如何测试 HTTP 客户端？

**答案**：

**测试策略**：

1. **单元测试 - Mock Transport**：
```python
from http_kit.testing import MockTransport, MockResponse

def test_get_user():
    transport = MockTransport([
        MockResponse(url="/users/1", json_data={"id": 1})
    ])
    client = HttpClient(transport=transport)
    response = client.get("/users/1")
    assert response.json()["id"] == 1
```

2. **使用 respx Mock**：
```python
import respx

@respx.mock
def test_with_respx():
    respx.get("/users").respond(json=[{"id": 1}])
    response = httpx.get("https://api.example.com/users")
    assert response.json() == [{"id": 1}]
```

3. **集成测试 - 真实 API**：
```python
@pytest.mark.integration
def test_real_api():
    response = httpx.get("https://api.example.com/health")
    assert response.status_code == 200
```

---

## 5. 连接池的作用？

**答案**：

**作用**：
- 复用 TCP 连接，避免重复握手
- 减少延迟和系统资源消耗
- 提高吞吐量

**对比**：
```
无连接池（每次新建连接）:
  请求1: [DNS] [TCP] [TLS] [请求] [响应] [关闭] = 200ms
  请求2: [DNS] [TCP] [TLS] [请求] [响应] [关闭] = 200ms
  总计: 400ms

有连接池（复用连接）:
  请求1: [DNS] [TCP] [TLS] [请求] [响应] = 200ms
  请求2:                   [请求] [响应] = 50ms
  总计: 250ms
```

**配置示例**：
```python
limits = httpx.Limits(
    max_keepalive_connections=20,  # 保持的连接数
    max_connections=100,            # 最大连接数
)
client = httpx.Client(limits=limits)
```

---

## 6. 如何传递 trace_id？

**答案**：

**实现方式**：

1. **使用 ContextVar（协程安全）**：
```python
from contextvars import ContextVar
import uuid

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def get_trace_id() -> str:
    tid = trace_id_var.get()
    if not tid:
        tid = str(uuid.uuid4())[:8]
        trace_id_var.set(tid)
    return tid
```

2. **在请求头中传递**：
```python
headers = {"X-Trace-Id": get_trace_id()}
response = client.get("/api", headers=headers)
```

3. **使用中间件自动注入**：
```python
class TracingMiddleware:
    def before_request(self, method, url, headers):
        headers["X-Trace-Id"] = get_trace_id()
        return headers
```

**用途**：
- 分布式追踪
- 日志关联
- 问题排查

---

## 7. 异步 HTTP 请求的优势？

**答案**：

**优势**：
1. **高并发**：单线程处理大量请求
2. **资源效率**：减少线程开销
3. **I/O 密集型场景**：等待网络时可处理其他任务

**对比**：
```python
# 同步 - 串行执行
def fetch_all_sync(urls):
    results = []
    for url in urls:
        results.append(httpx.get(url))  # 阻塞等待
    return results
# 100 个请求，每个 100ms = 10 秒

# 异步 - 并发执行
async def fetch_all_async(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)
# 100 个请求，并发执行 = ~100ms
```

**适用场景**：
- API 聚合
- 爬虫
- 微服务调用
- WebSocket

---

## 8. 如何处理大文件下载？

**答案**：

**流式下载**：
```python
import httpx

def download_file(url: str, path: str):
    with httpx.stream("GET", url) as response:
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
```

**异步流式下载**：
```python
async def download_file_async(url: str, path: str):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            async with aiofiles.open(path, "wb") as f:
                async for chunk in response.aiter_bytes(8192):
                    await f.write(chunk)
```

**进度显示**：
```python
def download_with_progress(url: str, path: str):
    with httpx.stream("GET", url) as response:
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(path, "wb") as f:
            for chunk in response.iter_bytes(8192):
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    progress = downloaded / total * 100
                    print(f"\rProgress: {progress:.1f}%", end="")
```

**关键点**：
- 使用 `stream()` 避免加载到内存
- 分块写入文件
- 检查 `Content-Length` 显示进度

---

## 9. 如何实现请求限流？

**答案**：

**令牌桶算法**：
```python
import time
import threading

class RateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = 1.0
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(1.0, self.tokens + elapsed * self.rate)

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

                wait = (1 - self.tokens) / self.rate

            time.sleep(wait)
```

**使用**：
```python
limiter = RateLimiter(rate=10)  # 10 RPS

for i in range(100):
    limiter.acquire()
    client.get("/api")
```

---

## 10. httpx 的超时配置有哪些？

**答案**：

```python
timeout = httpx.Timeout(
    connect=5.0,   # TCP 连接超时
    read=10.0,     # 读取响应超时
    write=10.0,    # 发送请求超时
    pool=5.0,      # 等待连接池超时
)

client = httpx.Client(timeout=timeout)
```

**各超时含义**：
- `connect`：建立 TCP 连接的时间
- `read`：从服务器读取数据的时间
- `write`：向服务器发送数据的时间
- `pool`：等待连接池可用连接的时间

**简单配置**：
```python
# 所有超时统一设置
client = httpx.Client(timeout=30.0)
```

