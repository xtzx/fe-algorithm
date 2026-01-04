# 可观测性

可观测性三大支柱：**日志（Logs）**、**指标（Metrics）**、**追踪（Traces）**

## 1. 结构化日志

### 1.1 为什么需要结构化日志

```python
# ❌ 传统日志 - 难以解析
logging.info(f"User {user_id} logged in from {ip}")

# ✅ 结构化日志 - 易于查询
logger.info("user_login", user_id=user_id, ip=ip, method="password")
```

### 1.2 使用 structlog

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()

# 使用
logger.info("request_handled", method="GET", path="/api/users", duration_ms=42)
# 输出: {"event": "request_handled", "method": "GET", "path": "/api/users", "duration_ms": 42, "level": "info", "timestamp": "2024-01-01T12:00:00Z"}
```

### 1.3 上下文绑定

```python
# 绑定 request_id，后续所有日志都会包含
structlog.contextvars.bind_contextvars(request_id="req-123")

logger.info("step_1")  # 包含 request_id
logger.info("step_2")  # 包含 request_id

# 清除
structlog.contextvars.clear_contextvars()
```

## 2. Prometheus 指标

### 2.1 指标类型

| 类型 | 用途 | 示例 |
|------|------|------|
| Counter | 只增不减的计数 | 请求总数、错误数 |
| Gauge | 可增可减的值 | 活跃连接数、队列长度 |
| Histogram | 分布统计 | 请求延迟分布 |
| Summary | 分位数统计 | P50、P99 延迟 |

### 2.2 定义指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# 请求延迟
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# 活跃连接
ACTIVE_CONNECTIONS = Gauge(
    "http_active_connections",
    "Active HTTP connections",
)
```

### 2.3 记录指标

```python
# 记录请求
REQUEST_COUNT.labels(method="GET", endpoint="/api/users", status_code="200").inc()
REQUEST_LATENCY.labels(method="GET", endpoint="/api/users").observe(0.05)

# 连接数
ACTIVE_CONNECTIONS.inc()  # 增加
ACTIVE_CONNECTIONS.dec()  # 减少
```

### 2.4 暴露指标

```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

## 3. 分布式追踪（OpenTelemetry）

### 3.1 概念

- **Trace**: 一次完整请求的追踪
- **Span**: 单个操作的时间段
- **Context**: 跨服务传播的上下文

### 3.2 配置

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 配置 Provider
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
```

### 3.3 使用

```python
# 创建 Span
with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order_id", 123)
    
    # 嵌套 Span
    with tracer.start_as_current_span("validate_order"):
        # 验证逻辑
        pass
    
    with tracer.start_as_current_span("save_to_db"):
        # 数据库操作
        pass
```

### 3.4 FastAPI 集成

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
```

## 4. 健康检查

### 4.1 存活检查（Liveness）

检查应用是否在运行：

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 4.2 就绪检查（Readiness）

检查应用是否准备好接收流量：

```python
@app.get("/health/ready")
async def readiness():
    # 检查依赖
    db_ok = await check_database()
    cache_ok = await check_cache()
    
    if db_ok and cache_ok:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Not ready")
```

### 4.3 Kubernetes 配置

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## 5. 可观测性工具栈

| 组件 | 推荐工具 |
|------|----------|
| 日志聚合 | ELK Stack / Loki |
| 指标监控 | Prometheus + Grafana |
| 分布式追踪 | Jaeger / Zipkin |
| APM | Datadog / New Relic |


