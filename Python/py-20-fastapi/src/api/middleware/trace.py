"""
Trace 中间件

提供:
- trace_id 生成
- 请求追踪
- 上下文管理
"""

import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# 使用 contextvars 存储 trace_id，支持异步上下文
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """获取当前请求的 trace_id"""
    return trace_id_var.get()


def generate_trace_id() -> str:
    """生成新的 trace_id"""
    return str(uuid.uuid4())


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Trace 中间件

    功能:
    1. 为每个请求生成唯一的 trace_id
    2. 如果请求头包含 X-Trace-ID，则使用该值
    3. 在响应头中返回 trace_id
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # 优先使用请求头中的 trace_id，否则生成新的
        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()

        # 设置到 context var
        token = trace_id_var.set(trace_id)

        try:
            # 将 trace_id 存入 request.state，方便其他地方访问
            request.state.trace_id = trace_id

            # 处理请求
            response = await call_next(request)

            # 在响应头中返回 trace_id
            response.headers["X-Trace-ID"] = trace_id

            return response
        finally:
            # 重置 context var
            trace_id_var.reset(token)

