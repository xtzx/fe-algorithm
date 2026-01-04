# 自定义异常

> 设计清晰的异常体系，让错误处理更优雅

## 为什么需要自定义异常

1. **语义清晰**：`UserNotFoundError` 比 `ValueError` 更明确
2. **分层处理**：不同层级捕获不同类型的异常
3. **携带信息**：自定义属性提供更多上下文
4. **统一管理**：应用级别的错误分类

---

## 基础：继承 Exception

```python
# 最简单的自定义异常
class MyAppError(Exception):
    """应用基础异常"""
    pass

# 使用
raise MyAppError("Something went wrong")
```

### 继承哪个类？

| 基类 | 场景 |
|------|------|
| `Exception` | 大多数情况（推荐）|
| `ValueError` | 值相关的错误 |
| `TypeError` | 类型相关的错误 |
| `RuntimeError` | 运行时错误 |

```python
# 继承更具体的异常
class InvalidAgeError(ValueError):
    """年龄值无效"""
    pass

class ConfigTypeError(TypeError):
    """配置类型错误"""
    pass
```

---

## 添加自定义属性

### 基础模式

```python
class APIError(Exception):
    """API 错误，包含状态码"""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

# 使用
try:
    raise APIError("Not found", status_code=404)
except APIError as e:
    print(f"Error: {e.message}")
    print(f"Status: {e.status_code}")
```

### 丰富的异常类

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class ErrorContext:
    """错误上下文"""
    timestamp: datetime
    request_id: str | None = None
    user_id: int | None = None
    extra: dict[str, Any] | None = None

class AppError(Exception):
    """应用异常基类"""

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        context: ErrorContext | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or ErrorContext(timestamp=datetime.now())

    def to_dict(self) -> dict:
        """转换为字典（用于 API 响应）"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "timestamp": self.context.timestamp.isoformat(),
            }
        }

# 使用
error = AppError(
    "User not found",
    code="USER_NOT_FOUND",
    context=ErrorContext(
        timestamp=datetime.now(),
        request_id="req-123",
        user_id=456,
    )
)
print(error.to_dict())
```

---

## 异常分层设计

### 推荐的层次结构

```python
# 应用基础异常
class AppError(Exception):
    """所有应用异常的基类"""
    pass

# 业务异常
class BusinessError(AppError):
    """业务逻辑错误"""
    pass

class ValidationError(BusinessError):
    """数据验证错误"""
    pass

class NotFoundError(BusinessError):
    """资源不存在"""
    pass

class PermissionError(BusinessError):
    """权限不足"""
    pass

# 基础设施异常
class InfrastructureError(AppError):
    """基础设施错误"""
    pass

class DatabaseError(InfrastructureError):
    """数据库错误"""
    pass

class CacheError(InfrastructureError):
    """缓存错误"""
    pass

class ExternalServiceError(InfrastructureError):
    """外部服务错误"""
    pass
```

### 使用分层异常

```python
# 业务层：抛出业务异常
def get_user(user_id: int) -> User:
    user = user_repo.find(user_id)
    if not user:
        raise NotFoundError(f"User {user_id} not found")
    return user

# API 层：统一处理
@app.exception_handler(BusinessError)
async def business_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )

@app.exception_handler(InfrastructureError)
async def infra_error_handler(request, exc):
    logger.error(f"Infrastructure error: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "Service temporarily unavailable"}
    )
```

---

## 完整示例：电商应用异常体系

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class ErrorCode(str, Enum):
    """错误码枚举"""
    # 通用
    UNKNOWN = "UNKNOWN_ERROR"
    VALIDATION = "VALIDATION_ERROR"

    # 用户相关
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_EXISTS = "USER_ALREADY_EXISTS"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"

    # 订单相关
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    INSUFFICIENT_STOCK = "INSUFFICIENT_STOCK"
    PAYMENT_FAILED = "PAYMENT_FAILED"

@dataclass
class ErrorDetail:
    """错误详情"""
    field: str | None = None
    value: Any = None
    constraint: str | None = None

class AppError(Exception):
    """应用异常基类"""

    default_code = ErrorCode.UNKNOWN
    default_message = "An error occurred"
    http_status = 500

    def __init__(
        self,
        message: str | None = None,
        code: ErrorCode | None = None,
        details: list[ErrorDetail] | None = None,
    ):
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details or []
        self.timestamp = datetime.now()
        super().__init__(self.message)

    def to_response(self) -> dict:
        """转换为 API 响应格式"""
        response = {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
            }
        }
        if self.details:
            response["error"]["details"] = [
                {k: v for k, v in detail.__dict__.items() if v is not None}
                for detail in self.details
            ]
        return response

# 业务异常
class ValidationError(AppError):
    """验证错误"""
    default_code = ErrorCode.VALIDATION
    default_message = "Validation failed"
    http_status = 400

class NotFoundError(AppError):
    """资源不存在"""
    default_message = "Resource not found"
    http_status = 404

class UserNotFoundError(NotFoundError):
    """用户不存在"""
    default_code = ErrorCode.USER_NOT_FOUND
    default_message = "User not found"

class OrderNotFoundError(NotFoundError):
    """订单不存在"""
    default_code = ErrorCode.ORDER_NOT_FOUND
    default_message = "Order not found"

class InsufficientStockError(AppError):
    """库存不足"""
    default_code = ErrorCode.INSUFFICIENT_STOCK
    default_message = "Insufficient stock"
    http_status = 400

    def __init__(self, product_id: int, requested: int, available: int):
        super().__init__(
            message=f"Insufficient stock for product {product_id}",
            details=[
                ErrorDetail(
                    field="quantity",
                    value=requested,
                    constraint=f"max: {available}"
                )
            ]
        )
        self.product_id = product_id
        self.requested = requested
        self.available = available

# 使用示例
def create_order(user_id: int, items: list[dict]) -> Order:
    # 验证用户
    user = get_user(user_id)
    if not user:
        raise UserNotFoundError()

    # 检查库存
    for item in items:
        stock = get_stock(item["product_id"])
        if stock < item["quantity"]:
            raise InsufficientStockError(
                product_id=item["product_id"],
                requested=item["quantity"],
                available=stock,
            )

    return Order(...)

# API 处理
try:
    order = create_order(user_id=123, items=[{"product_id": 1, "quantity": 10}])
except AppError as e:
    return JSONResponse(
        status_code=e.http_status,
        content=e.to_response()
    )
```

---

## 何时抛出异常 vs 返回错误

### 使用异常

```python
# ✅ 异常情况：真正的错误，调用者必须处理
def get_user(user_id: int) -> User:
    user = db.find(user_id)
    if not user:
        raise UserNotFoundError(f"User {user_id} not found")
    return user
```

### 使用返回值

```python
# ✅ 正常情况：可能没有结果是预期的
def find_user(user_id: int) -> User | None:
    return db.find(user_id)

# ✅ 使用 Result 模式
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass
class Result(Generic[T, E]):
    value: T | None = None
    error: E | None = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        return cls(value=value)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        return cls(error=error)

def validate_age(age: int) -> Result[int, str]:
    if age < 0:
        return Result.err("Age cannot be negative")
    if age > 150:
        return Result.err("Age seems unrealistic")
    return Result.ok(age)

# 使用
result = validate_age(25)
if result.is_ok:
    print(f"Valid age: {result.value}")
else:
    print(f"Invalid: {result.error}")
```

### 选择指南

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 程序无法继续 | 异常 | 强制调用者处理 |
| 可预期的"无结果" | 返回 None | 属于正常流程 |
| 多个可能的错误 | 异常或 Result | 根据复杂度选择 |
| 验证多个字段 | 异常列表或 Result | 收集所有错误 |
| 第三方库边界 | 异常 | 符合 Python 惯例 |

---

## 与 Pydantic 配合

```python
from pydantic import BaseModel, field_validator, ValidationError as PydanticValidationError

class UserCreate(BaseModel):
    name: str
    email: str
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v

# 转换 Pydantic 异常
def create_user(data: dict) -> User:
    try:
        user_data = UserCreate(**data)
    except PydanticValidationError as e:
        # 转换为应用异常
        details = [
            ErrorDetail(
                field=err["loc"][0] if err["loc"] else None,
                value=err.get("input"),
                constraint=err["msg"],
            )
            for err in e.errors()
        ]
        raise ValidationError("Invalid user data", details=details) from e

    return User(**user_data.model_dump())
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 继承 `BaseException` | 会被 `except Exception` 遗漏 | 继承 `Exception` |
| 忘记调用 `super().__init__` | 异常信息丢失 | 始终调用父类初始化 |
| 过度设计异常层次 | 难以维护 | 保持简单，按需扩展 |
| 异常类名不以 Error 结尾 | 不符合惯例 | 使用 `XxxError` 命名 |
| 异常信息硬编码 | 难以国际化 | 使用错误码 + 消息 |

---

## 小结

1. **继承 Exception**：除非有特殊需求
2. **添加有用属性**：错误码、字段、时间戳等
3. **设计层次结构**：便于分层捕获和处理
4. **与框架配合**：统一的异常处理中间件
5. **文档化**：说明何时抛出、如何处理

