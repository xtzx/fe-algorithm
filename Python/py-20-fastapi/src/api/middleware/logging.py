"""
请求日志中间件

提供:
- 请求/响应日志
- 耗时统计
- 结构化日志
"""

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.middleware.trace import get_trace_id

logger = logging.getLogger("api.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件

    记录:
    - 请求方法和路径
    - 响应状态码
    - 请求耗时
    - trace_id
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # 记录开始时间
        start_time = time.perf_counter()

        # 获取 trace_id
        trace_id = get_trace_id()

        # 记录请求信息
        logger.info(
            f"[{trace_id[:8]}] --> {request.method} {request.url.path}",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        # 处理请求
        response = await call_next(request)

        # 计算耗时
        process_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒

        # 记录响应信息
        log_method = logger.info if response.status_code < 400 else logger.warning
        log_method(
            f"[{trace_id[:8]}] <-- {request.method} {request.url.path} "
            f"{response.status_code} {process_time:.2f}ms",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time, 2),
            },
        )

        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        return response


def setup_logging(level: int = logging.INFO):
    """
    配置日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

