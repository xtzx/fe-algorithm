"""
异常处理

提供:
- 自定义异常类
- 统一错误响应格式
- 异常处理器注册
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.middleware.trace import get_trace_id


# ==================== 错误响应模型 ====================


class ErrorDetail(BaseModel):
    """错误详情"""

    field: str | None = None
    message: str
    code: str | None = None


class ErrorResponse(BaseModel):
    """统一错误响应格式"""

    success: bool = False
    error_code: str
    message: str
    details: list[ErrorDetail] | None = None
    trace_id: str | None = None


# ==================== 自定义异常 ====================


class AppException(Exception):
    """应用基础异常"""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: list[ErrorDetail] | None = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class NotFoundException(AppException):
    """资源不存在异常"""

    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            error_code="NOT_FOUND",
            message=f"{resource} {resource_id} 不存在",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class UnauthorizedException(AppException):
    """未授权异常"""

    def __init__(self, message: str = "未授权访问"):
        super().__init__(
            error_code="UNAUTHORIZED",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenException(AppException):
    """禁止访问异常"""

    def __init__(self, message: str = "没有权限执行此操作"):
        super().__init__(
            error_code="FORBIDDEN",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class ValidationException(AppException):
    """验证异常"""

    def __init__(self, message: str, details: list[ErrorDetail] | None = None):
        super().__init__(
            error_code="VALIDATION_ERROR",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ConflictException(AppException):
    """冲突异常（如资源已存在）"""

    def __init__(self, message: str):
        super().__init__(
            error_code="CONFLICT",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
        )


class RateLimitException(AppException):
    """请求频率限制异常"""

    def __init__(self, message: str = "请求过于频繁，请稍后再试"):
        super().__init__(
            error_code="RATE_LIMIT_EXCEEDED",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


class InternalServerException(AppException):
    """内部服务器错误"""

    def __init__(self, message: str = "服务器内部错误"):
        super().__init__(
            error_code="INTERNAL_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ==================== 异常处理器 ====================


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """应用异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            trace_id=get_trace_id(),
        ).model_dump(),
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """HTTPException 处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            trace_id=get_trace_id(),
        ).model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """请求验证异常处理器"""
    details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        details.append(
            ErrorDetail(
                field=field,
                message=error["msg"],
                code=error["type"],
            )
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="请求参数验证失败",
            details=details,
            trace_id=get_trace_id(),
        ).model_dump(),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """通用异常处理器（兜底）"""
    # 生产环境不暴露详细错误信息
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="服务器内部错误",
            trace_id=get_trace_id(),
        ).model_dump(),
    )


def register_exception_handlers(app: FastAPI):
    """注册所有异常处理器"""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    # 注意：生产环境建议启用通用异常处理器
    # app.add_exception_handler(Exception, generic_exception_handler)

