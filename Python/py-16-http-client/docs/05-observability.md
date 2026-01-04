# 可观测性

> 请求日志、trace_id 传递、计时统计

## 1. 为什么需要可观测性

- 调试问题
- 性能分析
- 监控告警
- 分布式追踪

## 2. 请求日志

### 基础日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("http_client")

def log_request(method: str, url: str):
    logger.info(f"→ {method} {url}")

def log_response(method: str, url: str, status: int, elapsed: float):
    logger.info(f"← {status} {method} {url} ({elapsed*1000:.1f}ms)")
```

### 使用中间件

```python
from http_kit import HttpClient
from http_kit.tracing import LoggingMiddleware

client = HttpClient(
    base_url="https://api.example.com",
    middlewares=[
        LoggingMiddleware(
            log_headers=True,
            log_body=True,
        ),
    ],
)
```

输出示例：
```
INFO: Request: GET /users
INFO:   Authorization: ***
INFO:   Accept: application/json
INFO: Response: 200 (45.2ms)
INFO:   Content-Type: application/json
INFO:   Body: [{"id": 1, "name": "Alice"}]
```

## 3. Trace ID 传递

### 什么是 Trace ID

分布式系统中追踪请求的唯一标识：

```
用户 → API Gateway → 服务A → 服务B → 数据库
        trace-id=abc123 传递到每个服务
```

### 使用 http_kit

```python
from http_kit import HttpClient
from http_kit.tracing import TracingMiddleware, set_trace_id

# 设置当前请求的 trace_id
set_trace_id("abc123")

client = HttpClient(
    base_url="https://api.example.com",
    middlewares=[TracingMiddleware()],
)

# 请求自动携带 X-Trace-Id 头
response = client.get("/users")
```

### 从上游接收 Trace ID

```python
from http_kit.tracing import set_trace_id, get_trace_id

# 在 Web 框架中接收
def handle_request(request):
    # 从请求头获取或生成新的
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())[:8]
    set_trace_id(trace_id)

    # 后续 HTTP 调用自动携带 trace_id
    response = client.get("/downstream-service")
```

### 手动实现

```python
import uuid
from contextvars import ContextVar

# 使用 ContextVar 存储 trace_id（协程安全）
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

def get_trace_id() -> str:
    trace_id = _trace_id.get()
    if not trace_id:
        trace_id = str(uuid.uuid4())[:8]
        _trace_id.set(trace_id)
    return trace_id

def set_trace_id(trace_id: str):
    _trace_id.set(trace_id)
```

## 4. 计时统计

### 使用 MetricsMiddleware

```python
from http_kit import HttpClient
from http_kit.tracing import MetricsMiddleware

metrics = MetricsMiddleware()

client = HttpClient(
    base_url="https://api.example.com",
    middlewares=[metrics],
)

# 发送一些请求
client.get("/users")
client.post("/orders", json={"item": "book"})
client.get("/products")

# 获取统计
summary = metrics.get_summary()
print(f"总请求数: {summary['total_requests']}")
print(f"平均延迟: {summary['avg_latency']*1000:.1f}ms")
print(f"错误率: {summary['error_rate']:.1f}%")
print(f"状态码分布: {summary['status_codes']}")
```

### 输出示例

```
总请求数: 3
平均延迟: 125.3ms
最小延迟: 45.2ms
最大延迟: 256.8ms
错误率: 0.0%
状态码分布: {200: 3}
```

## 5. 结构化日志

使用 JSON 格式便于分析：

```python
import json
import logging
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # 添加额外字段
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "url"):
            log_data["url"] = record.url
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "elapsed"):
            log_data["elapsed_ms"] = round(record.elapsed * 1000, 2)

        return json.dumps(log_data)

# 配置
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("http_kit")
logger.addHandler(handler)
```

输出：
```json
{"timestamp": "2024-01-15T10:30:00.123456", "level": "INFO", "message": "HTTP Request", "logger": "http_kit", "trace_id": "abc123", "method": "GET", "url": "/users", "status_code": 200, "elapsed_ms": 45.23}
```

## 6. 敏感信息处理

### 隐藏敏感头

```python
SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}

def sanitize_headers(headers: dict) -> dict:
    """隐藏敏感请求头"""
    return {
        k: "***" if k.lower() in SENSITIVE_HEADERS else v
        for k, v in headers.items()
    }
```

### 隐藏敏感参数

```python
SENSITIVE_PARAMS = {"password", "token", "secret", "api_key"}

def sanitize_params(params: dict) -> dict:
    """隐藏敏感参数"""
    return {
        k: "***" if k.lower() in SENSITIVE_PARAMS else v
        for k, v in params.items()
    }
```

## 7. 与 JS 对比

```python
# Python
from http_kit.tracing import TracingMiddleware

client = HttpClient(middlewares=[TracingMiddleware()])
```

```javascript
// JavaScript with axios
axios.interceptors.request.use((config) => {
  config.headers['X-Trace-Id'] = getTraceId();
  console.log(`→ ${config.method} ${config.url}`);
  return config;
});

axios.interceptors.response.use((response) => {
  console.log(`← ${response.status} ${response.config.url}`);
  return response;
});
```

## 8. 生产环境配置

```python
import logging
from http_kit import HttpClient
from http_kit.tracing import TracingMiddleware, MetricsMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 创建中间件
tracing = TracingMiddleware(
    trace_id_header="X-Trace-Id",
    log_level=logging.INFO,
)

metrics = MetricsMiddleware(
    max_history=10000,
)

# 创建客户端
client = HttpClient(
    base_url="https://api.example.com",
    middlewares=[tracing, metrics],
)

# 定期导出指标
def export_metrics():
    summary = metrics.get_summary()
    # 发送到监控系统...
```

## 小结

| 概念 | 要点 |
|------|------|
| 请求日志 | 记录方法、URL、状态码、耗时 |
| Trace ID | 使用 ContextVar，跨服务传递 |
| 计时统计 | 平均、最大、最小延迟 |
| 结构化日志 | JSON 格式便于分析 |
| 敏感信息 | 隐藏 Authorization 等 |

