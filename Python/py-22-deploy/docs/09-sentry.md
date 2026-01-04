# Sentry 错误监控

> 生产环境错误追踪与性能监控

## 什么是 Sentry

Sentry 是开源的错误追踪和性能监控平台：
- 自动捕获异常
- 错误聚合与去重
- 上下文信息收集
- 性能监控（APM）
- 告警通知

---

## 安装配置

### 安装

```bash
pip install sentry-sdk[fastapi]
```

### 基础配置

```python
# config.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration

sentry_sdk.init(
    dsn="https://xxx@sentry.io/xxx",  # 从 Sentry 项目获取

    # 环境和版本
    environment="production",  # 或 staging, development
    release="myapp@1.0.0",     # 版本号

    # 采样率
    traces_sample_rate=0.1,    # 10% 性能追踪
    profiles_sample_rate=0.1,  # 10% 性能分析

    # 集成
    integrations=[
        FastApiIntegration(),
        SqlalchemyIntegration(),
        RedisIntegration(),
    ],

    # 数据过滤
    send_default_pii=False,    # 不发送个人信息
)
```

### 从环境变量配置

```python
import os
import sentry_sdk

SENTRY_DSN = os.getenv("SENTRY_DSN")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
VERSION = os.getenv("VERSION", "unknown")

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=ENVIRONMENT,
        release=f"myapp@{VERSION}",
        traces_sample_rate=0.1 if ENVIRONMENT == "production" else 1.0,
    )
```

---

## FastAPI 集成

### 基础使用

```python
from fastapi import FastAPI
import sentry_sdk

sentry_sdk.init(dsn="...")

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/error")
async def trigger_error():
    # 这个错误会自动发送到 Sentry
    raise ValueError("Something went wrong!")
```

### 手动捕获

```python
import sentry_sdk

# 捕获异常
try:
    risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    # 可以继续处理或重新抛出

# 捕获消息
sentry_sdk.capture_message("Something happened", level="info")
```

### 添加上下文

```python
import sentry_sdk

# 设置用户信息
sentry_sdk.set_user({
    "id": user.id,
    "email": user.email,
    "username": user.name,
})

# 设置标签（可搜索）
sentry_sdk.set_tag("feature", "checkout")
sentry_sdk.set_tag("customer_type", "premium")

# 设置额外数据
sentry_sdk.set_extra("order_id", order.id)
sentry_sdk.set_extra("items_count", len(cart.items))

# 设置上下文（结构化数据）
sentry_sdk.set_context("order", {
    "id": order.id,
    "total": order.total,
    "items": len(order.items),
})
```

### 使用 Scope

```python
import sentry_sdk

# 临时上下文
with sentry_sdk.push_scope() as scope:
    scope.set_tag("request_id", request_id)
    scope.set_extra("payload", payload)

    try:
        process_request(payload)
    except Exception as e:
        sentry_sdk.capture_exception(e)

# scope 结束后上下文自动清除
```

---

## 中间件集成

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk
import uuid

class SentryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 生成请求 ID
        request_id = str(uuid.uuid4())

        with sentry_sdk.push_scope() as scope:
            # 添加请求上下文
            scope.set_tag("request_id", request_id)
            scope.set_tag("path", request.url.path)
            scope.set_tag("method", request.method)

            # 添加用户信息（如果有）
            if hasattr(request.state, "user"):
                scope.set_user({
                    "id": request.state.user.id,
                    "email": request.state.user.email,
                })

            try:
                response = await call_next(request)
                return response
            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise

app = FastAPI()
app.add_middleware(SentryMiddleware)
```

---

## 性能监控

### 自动追踪

```python
# FastAPI 集成自动追踪请求
sentry_sdk.init(
    dsn="...",
    traces_sample_rate=0.1,  # 10% 采样
    integrations=[FastApiIntegration()],
)

# 数据库查询、HTTP 请求等自动追踪
```

### 自定义事务

```python
import sentry_sdk

@app.post("/process-order")
async def process_order(order_id: int):
    # 开始事务
    with sentry_sdk.start_transaction(
        op="task",
        name="process_order"
    ) as transaction:

        # 子 span
        with sentry_sdk.start_span(op="db.query", description="Get order"):
            order = await get_order(order_id)

        with sentry_sdk.start_span(op="http", description="Payment API"):
            payment = await process_payment(order)

        with sentry_sdk.start_span(op="task", description="Send notification"):
            await send_notification(order)

        return {"status": "completed"}
```

### Span 装饰器

```python
from sentry_sdk import trace

@trace
async def get_user_data(user_id: int):
    # 自动创建 span
    return await db.get_user(user_id)

@trace
async def send_email(to: str, subject: str):
    # 自动创建 span
    await email_client.send(to, subject)
```

---

## 数据过滤

### 过滤敏感数据

```python
import sentry_sdk
from sentry_sdk.types import Event, Hint

def before_send(event: Event, hint: Hint) -> Event | None:
    # 过滤特定异常
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, IgnoredException):
            return None

    # 过滤敏感数据
    if "request" in event:
        request = event["request"]

        # 移除敏感 headers
        if "headers" in request:
            sensitive_headers = ["authorization", "cookie", "x-api-key"]
            for header in sensitive_headers:
                request["headers"].pop(header, None)

        # 移除敏感 body 字段
        if "data" in request:
            for field in ["password", "credit_card", "ssn"]:
                if field in request["data"]:
                    request["data"][field] = "[FILTERED]"

    return event

sentry_sdk.init(
    dsn="...",
    before_send=before_send,
)
```

### 过滤特定错误

```python
def before_send(event, hint):
    # 忽略 404 错误
    if "exception" in event:
        exc = event["exception"]["values"][0]
        if exc.get("type") == "HTTPException" and exc.get("value", "").startswith("404"):
            return None

    return event
```

---

## 告警配置

### Sentry 告警规则

在 Sentry 后台配置：

1. **Issue Alerts**（问题告警）
   - 新问题出现时
   - 问题重新出现时
   - 问题发生频率超过阈值

2. **Metric Alerts**（指标告警）
   - 错误率超过 5%
   - 响应时间 P95 > 1秒
   - 事务吞吐量下降

### 告警通知

支持的通知渠道：
- Slack
- Email
- PagerDuty
- Microsoft Teams
- Webhook

```python
# Webhook 示例
# POST /webhook/sentry
@app.post("/webhook/sentry")
async def sentry_webhook(payload: dict):
    event_type = payload.get("action")

    if event_type == "triggered":
        # 告警触发
        alert_name = payload.get("data", {}).get("metric_alert", {}).get("alert_rule", {}).get("name")
        await send_notification(f"Sentry Alert: {alert_name}")

    return {"status": "ok"}
```

---

## 生产环境最佳实践

### 配置示例

```python
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

def init_sentry():
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return

    environment = os.getenv("ENVIRONMENT", "development")

    # 根据环境调整采样率
    traces_sample_rate = {
        "production": 0.1,
        "staging": 0.5,
        "development": 1.0,
    }.get(environment, 0.1)

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=os.getenv("VERSION", "unknown"),

        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=traces_sample_rate,

        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            RedisIntegration(),
            CeleryIntegration(),
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],

        # 性能
        enable_tracing=True,

        # 隐私
        send_default_pii=False,

        # 数据处理
        before_send=before_send,
        before_send_transaction=before_send_transaction,

        # 忽略的错误
        ignore_errors=[
            KeyboardInterrupt,
            SystemExit,
        ],
    )
```

### 健康检查排除

```python
def before_send_transaction(event, hint):
    # 排除健康检查
    if event.get("transaction") in ["/health", "/ready", "/metrics"]:
        return None
    return event
```

---

## 本地开发

### 禁用 Sentry

```python
# 开发环境不上报
import os

if os.getenv("ENVIRONMENT") != "development":
    sentry_sdk.init(dsn="...")
```

### 调试模式

```python
sentry_sdk.init(
    dsn="...",
    debug=True,  # 打印调试信息
)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| DSN 泄露 | 硬编码在代码中 | 使用环境变量 |
| 采样率太高 | 产生大量费用 | 生产环境 0.1-0.2 |
| 敏感数据上报 | 隐私问题 | 配置 before_send 过滤 |
| 忽略环境区分 | 开发错误混入 | 设置 environment |
| 版本不标记 | 不知道哪个版本出问题 | 设置 release |

---

## 小结

1. **安装**：`sentry-sdk[fastapi]`
2. **配置**：DSN、environment、release
3. **上下文**：user、tags、extra、context
4. **性能监控**：traces_sample_rate
5. **数据过滤**：before_send
6. **告警**：配置规则和通知渠道

