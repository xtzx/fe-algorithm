"""
可观测性模块

支持：
- 请求日志
- trace_id 传递
- 计时统计
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import httpx

# 当前 trace_id 的上下文变量
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def get_trace_id() -> str:
    """获取当前 trace_id，如果没有则生成一个"""
    trace_id = current_trace_id.get()
    if trace_id is None:
        trace_id = str(uuid.uuid4())[:8]
        current_trace_id.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    """设置当前 trace_id"""
    current_trace_id.set(trace_id)


class Middleware(ABC):
    """中间件基类"""

    @abstractmethod
    def before_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, str]:
        """
        请求前处理

        Args:
            method: HTTP 方法
            url: 请求 URL
            headers: 请求头

        Returns:
            修改后的请求头
        """
        pass

    @abstractmethod
    def after_request(
        self,
        method: str,
        url: str,
        response: httpx.Response,
        elapsed: float,
    ) -> None:
        """
        请求后处理

        Args:
            method: HTTP 方法
            url: 请求 URL
            response: HTTP 响应
            elapsed: 耗时（秒）
        """
        pass


class TracingMiddleware(Middleware):
    """
    追踪中间件

    自动添加 trace_id 到请求头，记录请求日志

    Example:
        ```python
        middleware = TracingMiddleware()
        client = HttpClient(middlewares=[middleware])
        ```
    """

    def __init__(
        self,
        trace_id_header: str = "X-Trace-Id",
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.trace_id_header = trace_id_header
        self.logger = logger or logging.getLogger("http_kit")
        self.log_level = log_level

    def before_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, str]:
        """添加 trace_id 到请求头"""
        trace_id = get_trace_id()
        headers = dict(headers)
        headers[self.trace_id_header] = trace_id

        self.logger.log(
            self.log_level,
            f"[{trace_id}] → {method} {url}",
        )

        return headers

    def after_request(
        self,
        method: str,
        url: str,
        response: httpx.Response,
        elapsed: float,
    ) -> None:
        """记录响应日志"""
        trace_id = get_trace_id()

        self.logger.log(
            self.log_level,
            f"[{trace_id}] ← {response.status_code} {method} {url} ({elapsed*1000:.1f}ms)",
        )


class LoggingMiddleware(Middleware):
    """
    日志中间件

    详细记录请求和响应

    Example:
        ```python
        middleware = LoggingMiddleware(log_headers=True, log_body=True)
        ```
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.DEBUG,
        log_headers: bool = False,
        log_body: bool = False,
        max_body_length: int = 1000,
    ) -> None:
        self.logger = logger or logging.getLogger("http_kit")
        self.log_level = log_level
        self.log_headers = log_headers
        self.log_body = log_body
        self.max_body_length = max_body_length

    def before_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, str]:
        """记录请求详情"""
        self.logger.log(self.log_level, f"Request: {method} {url}")

        if self.log_headers:
            for key, value in headers.items():
                # 隐藏敏感头
                if key.lower() in ("authorization", "x-api-key", "cookie"):
                    value = "***"
                self.logger.log(self.log_level, f"  {key}: {value}")

        return headers

    def after_request(
        self,
        method: str,
        url: str,
        response: httpx.Response,
        elapsed: float,
    ) -> None:
        """记录响应详情"""
        self.logger.log(
            self.log_level,
            f"Response: {response.status_code} ({elapsed*1000:.1f}ms)",
        )

        if self.log_headers:
            for key, value in response.headers.items():
                self.logger.log(self.log_level, f"  {key}: {value}")

        if self.log_body:
            body = response.text
            if len(body) > self.max_body_length:
                body = body[: self.max_body_length] + "..."
            self.logger.log(self.log_level, f"  Body: {body}")


@dataclass
class RequestMetrics:
    """请求指标"""

    method: str
    url: str
    status_code: int
    elapsed: float
    timestamp: float = field(default_factory=time.time)


class MetricsMiddleware(Middleware):
    """
    指标中间件

    收集请求指标用于监控

    Example:
        ```python
        middleware = MetricsMiddleware()
        client = HttpClient(middlewares=[middleware])

        # 获取指标
        metrics = middleware.get_metrics()
        ```
    """

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._metrics: list[RequestMetrics] = []

    def before_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> dict[str, str]:
        return headers

    def after_request(
        self,
        method: str,
        url: str,
        response: httpx.Response,
        elapsed: float,
    ) -> None:
        """记录指标"""
        metric = RequestMetrics(
            method=method,
            url=url,
            status_code=response.status_code,
            elapsed=elapsed,
        )

        self._metrics.append(metric)

        # 限制历史记录数量
        if len(self._metrics) > self.max_history:
            self._metrics = self._metrics[-self.max_history :]

    def get_metrics(self) -> list[RequestMetrics]:
        """获取所有指标"""
        return list(self._metrics)

    def get_summary(self) -> dict[str, Any]:
        """获取指标摘要"""
        if not self._metrics:
            return {
                "total_requests": 0,
                "avg_latency": 0,
                "error_rate": 0,
            }

        total = len(self._metrics)
        errors = sum(1 for m in self._metrics if m.status_code >= 400)
        total_latency = sum(m.elapsed for m in self._metrics)

        return {
            "total_requests": total,
            "avg_latency": total_latency / total,
            "min_latency": min(m.elapsed for m in self._metrics),
            "max_latency": max(m.elapsed for m in self._metrics),
            "error_rate": errors / total * 100,
            "status_codes": self._count_status_codes(),
        }

    def _count_status_codes(self) -> dict[int, int]:
        """统计状态码分布"""
        counts: dict[int, int] = {}
        for m in self._metrics:
            counts[m.status_code] = counts.get(m.status_code, 0) + 1
        return counts

    def clear(self) -> None:
        """清空指标"""
        self._metrics.clear()

