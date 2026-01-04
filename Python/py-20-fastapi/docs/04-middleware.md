# 中间件

## 概述

中间件在请求处理前后执行，用于：

1. **CORS** - 跨域资源共享
2. **日志** - 请求/响应日志
3. **追踪** - trace_id
4. **认证** - 全局认证
5. **性能** - 响应时间统计

## 1. CORS 中间件

### 1.1 基础配置

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.2 动态 CORS

```python
def get_cors_origins() -> list[str]:
    """从配置获取允许的源"""
    settings = get_settings()
    return settings.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Request-ID"],
    max_age=600,  # 预检请求缓存时间（秒）
)
```

## 2. 自定义中间件

### 2.1 基于函数的中间件

```python
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头"""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response
```

### 2.2 基于类的中间件

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class TimingMiddleware(BaseHTTPMiddleware):
    """计时中间件"""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

app.add_middleware(TimingMiddleware)
```

## 3. 请求日志中间件

```python
import logging
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.request")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # 请求开始
        start_time = time.perf_counter()

        # 记录请求
        logger.info(
            f"--> {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client": request.client.host if request.client else "unknown",
            }
        )

        # 处理请求
        response = await call_next(request)

        # 计算耗时
        process_time = (time.perf_counter() - start_time) * 1000

        # 记录响应
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            f"<-- {request.method} {request.url.path} {response.status_code} {process_time:.2f}ms",
            extra={
                "status_code": response.status_code,
                "process_time_ms": round(process_time, 2),
            }
        )

        return response
```

## 4. Trace 中间件

```python
import uuid
from contextvars import ContextVar

# 使用 contextvars 存储 trace_id
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def get_trace_id() -> str:
    """获取当前请求的 trace_id"""
    return trace_id_var.get()

class TraceMiddleware(BaseHTTPMiddleware):
    """Trace ID 中间件"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # 优先使用请求头中的 trace_id
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())

        # 设置到 context var
        token = trace_id_var.set(trace_id)

        try:
            # 存入 request.state
            request.state.trace_id = trace_id

            response = await call_next(request)

            # 响应头返回 trace_id
            response.headers["X-Trace-ID"] = trace_id

            return response
        finally:
            trace_id_var.reset(token)
```

### 在日志中使用 trace_id

```python
import logging

class TraceFilter(logging.Filter):
    """添加 trace_id 到日志"""

    def filter(self, record):
        record.trace_id = get_trace_id() or "no-trace"
        return True

# 配置日志格式
logging.basicConfig(
    format="%(asctime)s [%(trace_id)s] %(levelname)s %(message)s"
)
logger = logging.getLogger()
logger.addFilter(TraceFilter())
```

## 5. 认证中间件

```python
from fastapi import HTTPException

class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""

    # 不需要认证的路径
    WHITELIST = ["/", "/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next) -> Response:
        # 白名单路径跳过认证
        if request.url.path in self.WHITELIST:
            return await call_next(request)

        # 检查 Authorization 头
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"},
            )

        # 验证 token
        try:
            token = auth_header.replace("Bearer ", "")
            user = verify_token(token)
            request.state.user = user
        except Exception:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"},
            )

        return await call_next(request)
```

## 6. 限流中间件

```python
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    """简单的限流中间件"""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"

        # 清理过期请求
        now = time.time()
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if now - t < self.window
        ]

        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
            )

        # 记录请求
        self.requests[client_ip].append(now)

        response = await call_next(request)

        # 添加限流头
        remaining = self.max_requests - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
```

## 7. 中间件执行顺序

```python
# 中间件按添加顺序的逆序执行
# 最后添加的最先执行

app.add_middleware(MiddlewareA)  # 第三执行
app.add_middleware(MiddlewareB)  # 第二执行
app.add_middleware(MiddlewareC)  # 第一执行

# 执行顺序：
# 请求：C -> B -> A -> Handler
# 响应：Handler -> A -> B -> C
```

### 推荐顺序

```python
# 1. CORS（最外层）
app.add_middleware(CORSMiddleware, ...)

# 2. 日志和追踪
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TraceMiddleware)

# 3. 认证
app.add_middleware(AuthMiddleware)

# 4. 限流
app.add_middleware(RateLimitMiddleware)
```

## 8. Starlette 内置中间件

```python
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Gzip 压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# HTTPS 重定向
app.add_middleware(HTTPSRedirectMiddleware)

# 可信主机
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"],
)
```

## Python vs JavaScript 对比

| 特性 | FastAPI | Express.js |
|------|---------|------------|
| 添加 | `app.add_middleware()` | `app.use()` |
| 类型 | 类或装饰器 | 函数 |
| 请求 | `request: Request` | `req` |
| 响应 | `call_next(request)` | `next()` |
| 顺序 | 逆序执行 | 顺序执行 |
| 跳过 | 早返回 | `next()` |

