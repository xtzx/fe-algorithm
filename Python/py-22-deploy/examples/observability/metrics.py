"""
Prometheus 指标

演示:
- 指标类型
- 指标注册
- FastAPI 集成
- 自定义指标
"""

import time
from functools import wraps
from typing import Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)


# ==================== 指标定义 ====================


# Counter - 只增不减的计数器
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# Gauge - 可增可减的值
ACTIVE_CONNECTIONS = Gauge(
    "http_active_connections",
    "Number of active HTTP connections",
)

IN_PROGRESS_REQUESTS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
)

# Histogram - 分布统计
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Summary - 分位数统计
REQUEST_SIZE = Summary(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
)

# Info - 静态信息
APP_INFO = Info(
    "app",
    "Application information",
)


# ==================== 指标收集 ====================


def setup_metrics(app_name: str, version: str):
    """设置应用信息指标"""
    APP_INFO.info({
        "name": app_name,
        "version": version,
    })


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """记录请求指标"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def track_in_progress(method: str, endpoint: str):
    """追踪进行中的请求（装饰器）"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            IN_PROGRESS_REQUESTS.labels(method=method, endpoint=endpoint).inc()
            try:
                return await func(*args, **kwargs)
            finally:
                IN_PROGRESS_REQUESTS.labels(method=method, endpoint=endpoint).dec()
        return wrapper
    return decorator


def time_request(method: str, endpoint: str):
    """计时装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                status_code = 200
                return result
            except Exception:
                status_code = 500
                raise
            finally:
                duration = time.perf_counter() - start
                record_request(method, endpoint, status_code, duration)
        return wrapper
    return decorator


# ==================== FastAPI 集成 ====================


def create_metrics_middleware():
    """创建 FastAPI 指标中间件"""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    
    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            method = request.method
            path = request.url.path
            
            # 跳过指标端点本身
            if path == "/metrics":
                return await call_next(request)
            
            IN_PROGRESS_REQUESTS.labels(method=method, endpoint=path).inc()
            ACTIVE_CONNECTIONS.inc()
            
            start = time.perf_counter()
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.perf_counter() - start
                
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code,
                ).inc()
                
                REQUEST_LATENCY.labels(
                    method=method,
                    endpoint=path,
                ).observe(duration)
                
                IN_PROGRESS_REQUESTS.labels(method=method, endpoint=path).dec()
                ACTIVE_CONNECTIONS.dec()
    
    return MetricsMiddleware


def metrics_endpoint():
    """创建 /metrics 端点"""
    from fastapi import Response
    
    def get_metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    return get_metrics


# ==================== 自定义业务指标 ====================


class BusinessMetrics:
    """业务指标"""
    
    def __init__(self):
        self.orders_created = Counter(
            "business_orders_created_total",
            "Total orders created",
            ["product_type"],
        )
        
        self.order_value = Histogram(
            "business_order_value",
            "Order value distribution",
            buckets=(10, 50, 100, 500, 1000, 5000),
        )
        
        self.active_users = Gauge(
            "business_active_users",
            "Number of active users",
        )
        
        self.cache_hit_rate = Gauge(
            "business_cache_hit_rate",
            "Cache hit rate",
            ["cache_name"],
        )
    
    def record_order(self, product_type: str, value: float):
        self.orders_created.labels(product_type=product_type).inc()
        self.order_value.observe(value)
    
    def set_active_users(self, count: int):
        self.active_users.set(count)
    
    def set_cache_hit_rate(self, cache_name: str, rate: float):
        self.cache_hit_rate.labels(cache_name=cache_name).set(rate)


# ==================== 使用示例 ====================


def demo_metrics():
    """演示指标使用"""
    
    # 设置应用信息
    setup_metrics("myapp", "1.0.0")
    
    # 记录请求
    record_request("GET", "/api/users", 200, 0.05)
    record_request("POST", "/api/orders", 201, 0.15)
    record_request("GET", "/api/users", 500, 0.02)
    
    # 业务指标
    biz = BusinessMetrics()
    biz.record_order("electronics", 299.99)
    biz.record_order("books", 29.99)
    biz.set_active_users(1234)
    biz.set_cache_hit_rate("user_cache", 0.85)
    
    # 打印指标
    print(generate_latest().decode())


if __name__ == "__main__":
    demo_metrics()


