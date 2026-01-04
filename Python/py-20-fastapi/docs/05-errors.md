# 错误处理

## 概述

FastAPI 提供多种错误处理方式：

1. **HTTPException** - 内置 HTTP 异常
2. **自定义异常** - 业务异常
3. **异常处理器** - 统一错误格式
4. **验证错误** - Pydantic 验证失败

## 1. HTTPException

### 1.1 基础用法

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="商品不存在",
        )
    return items_db[item_id]
```

### 1.2 带响应头

```python
@app.get("/protected")
async def protected_route():
    raise HTTPException(
        status_code=401,
        detail="需要认证",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

### 1.3 复杂的 detail

```python
raise HTTPException(
    status_code=400,
    detail={
        "code": "INVALID_INPUT",
        "message": "输入无效",
        "errors": [
            {"field": "email", "message": "邮箱格式错误"},
            {"field": "age", "message": "年龄必须大于 0"},
        ],
    },
)
```

## 2. 自定义异常

### 2.1 定义异常类

```python
class AppException(Exception):
    """应用基础异常"""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 400,
        details: list | None = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class NotFoundException(AppException):
    """资源不存在"""

    def __init__(self, resource: str, resource_id):
        super().__init__(
            error_code="NOT_FOUND",
            message=f"{resource} {resource_id} 不存在",
            status_code=404,
        )


class UnauthorizedException(AppException):
    """未授权"""

    def __init__(self, message: str = "未授权访问"):
        super().__init__(
            error_code="UNAUTHORIZED",
            message=message,
            status_code=401,
        )


class ForbiddenException(AppException):
    """禁止访问"""

    def __init__(self, message: str = "没有权限"):
        super().__init__(
            error_code="FORBIDDEN",
            message=message,
            status_code=403,
        )


class ValidationException(AppException):
    """验证失败"""

    def __init__(self, message: str, details: list | None = None):
        super().__init__(
            error_code="VALIDATION_ERROR",
            message=message,
            status_code=422,
            details=details,
        )
```

### 2.2 使用自定义异常

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.get_user(user_id)
    if not user:
        raise NotFoundException("用户", user_id)
    return user

@app.delete("/items/{item_id}")
async def delete_item(item_id: int, current_user: User = Depends(get_current_user)):
    item = db.get_item(item_id)
    if item.owner_id != current_user.id:
        raise ForbiddenException("只能删除自己的商品")
    db.delete_item(item_id)
```

## 3. 异常处理器

### 3.1 自定义异常处理器

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )
```

### 3.2 覆盖 HTTPException 处理器

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
        },
        headers=exc.headers,
    )
```

### 3.3 验证错误处理器

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "请求参数验证失败",
            "details": errors,
        },
    )
```

### 3.4 通用异常处理器

```python
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # 记录错误日志
    logger.exception(f"Unhandled exception: {exc}")

    # 生产环境不暴露详细错误
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
        },
    )
```

## 4. 统一错误格式

### 4.1 错误响应模型

```python
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    field: str | None = None
    message: str
    code: str | None = None

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    details: list[ErrorDetail] | None = None
    trace_id: str | None = None
```

### 4.2 注册所有处理器

```python
def register_exception_handlers(app: FastAPI):
    """注册所有异常处理器"""

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
                trace_id=get_trace_id(),
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                trace_id=get_trace_id(),
            ).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        details = [
            ErrorDetail(
                field=".".join(str(loc) for loc in e["loc"]),
                message=e["msg"],
                code=e["type"],
            )
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="请求参数验证失败",
                details=details,
                trace_id=get_trace_id(),
            ).model_dump(),
        )
```

## 5. 业务错误码

```python
from enum import Enum

class ErrorCode(str, Enum):
    # 认证相关 (1xxx)
    UNAUTHORIZED = "1001"
    INVALID_TOKEN = "1002"
    TOKEN_EXPIRED = "1003"

    # 资源相关 (2xxx)
    NOT_FOUND = "2001"
    ALREADY_EXISTS = "2002"

    # 权限相关 (3xxx)
    FORBIDDEN = "3001"
    INSUFFICIENT_PERMISSION = "3002"

    # 验证相关 (4xxx)
    VALIDATION_ERROR = "4001"
    INVALID_FORMAT = "4002"

    # 业务相关 (5xxx)
    BUSINESS_ERROR = "5001"
    INSUFFICIENT_BALANCE = "5002"

class BusinessException(AppException):
    """业务异常"""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(
            error_code=code.value,
            message=message,
            status_code=400,
        )

# 使用
raise BusinessException(
    ErrorCode.INSUFFICIENT_BALANCE,
    "余额不足，当前余额: 100，需要: 200",
)
```

## 6. 错误日志

```python
import logging

logger = logging.getLogger("api.error")

@app.exception_handler(Exception)
async def log_exception_handler(request: Request, exc: Exception):
    # 获取 trace_id
    trace_id = getattr(request.state, "trace_id", "unknown")

    # 记录详细日志
    logger.error(
        f"[{trace_id}] Unhandled exception",
        exc_info=True,
        extra={
            "trace_id": trace_id,
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
            "trace_id": trace_id,
        },
    )
```

## 7. 文档中的错误响应

```python
@app.get(
    "/users/{user_id}",
    responses={
        200: {"model": UserResponse, "description": "成功"},
        404: {
            "model": ErrorResponse,
            "description": "用户不存在",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "NOT_FOUND",
                        "message": "用户 123 不存在",
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "验证失败",
        },
    },
)
async def get_user(user_id: int):
    ...
```

## Python vs JavaScript 对比

| 特性 | FastAPI | Express.js |
|------|---------|------------|
| 抛出 | `raise HTTPException()` | `throw new Error()` |
| 处理器 | `@app.exception_handler()` | `app.use((err, req, res, next))` |
| 验证错误 | `RequestValidationError` | 手动处理 |
| 状态码 | `status.HTTP_404_NOT_FOUND` | `404` |
| 响应 | `JSONResponse()` | `res.json()` |

