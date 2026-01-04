# Pydantic 集成

## 概述

FastAPI 深度集成 Pydantic v2，提供：

1. **请求验证** - 自动验证请求参数
2. **响应序列化** - 自动转换响应数据
3. **文档生成** - 自动生成 API 文档

## 1. 请求验证

### 1.1 基础模型

```python
from pydantic import BaseModel, Field
from datetime import datetime

class UserCreate(BaseModel):
    """创建用户请求"""
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="用户名",
    )
    email: str = Field(..., description="邮箱")
    password: str = Field(
        ...,
        min_length=8,
        description="密码",
    )
    age: int | None = Field(None, ge=0, le=150)

@app.post("/users")
async def create_user(user: UserCreate):
    # user 已经过验证
    return {"username": user.username}
```

### 1.2 嵌套模型

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "中国"

class Company(BaseModel):
    name: str
    address: Address

class User(BaseModel):
    name: str
    company: Company | None = None

@app.post("/users")
async def create_user(user: User):
    return user
```

### 1.3 列表和字典

```python
class OrderItem(BaseModel):
    product_id: int
    quantity: int = Field(ge=1)

class Order(BaseModel):
    items: list[OrderItem]
    metadata: dict[str, str] = {}

@app.post("/orders")
async def create_order(order: Order):
    return {"item_count": len(order.items)}
```

### 1.4 字段验证

```python
from pydantic import field_validator, model_validator

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("无效的邮箱格式")
        return v.lower()

    @model_validator(mode="after")
    def check_passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("两次密码不一致")
        return self
```

## 2. 响应序列化

### 2.1 响应模型

```python
class UserResponse(BaseModel):
    """用户响应（不包含敏感信息）"""
    id: int
    username: str
    email: str
    created_at: datetime

    # 允许从 ORM 对象创建
    model_config = {"from_attributes": True}

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    # 即使包含 password，也不会返回
    user = db.get_user(user_id)
    return user
```

### 2.2 不同场景的响应

```python
# 创建时返回完整信息
class UserCreateResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

# 列表返回简要信息
class UserListItem(BaseModel):
    id: int
    username: str

# 详情返回完整信息
class UserDetail(BaseModel):
    id: int
    username: str
    email: str
    profile: dict
    created_at: datetime
    updated_at: datetime | None

@app.get("/users", response_model=list[UserListItem])
async def list_users():
    return users

@app.get("/users/{user_id}", response_model=UserDetail)
async def get_user(user_id: int):
    return user
```

### 2.3 排除和包含字段

```python
@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    response_model_exclude={"email"},  # 排除 email
)
async def get_user_public(user_id: int):
    return user

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    response_model_include={"id", "username"},  # 只包含这些
)
async def get_user_minimal(user_id: int):
    return user

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    response_model_exclude_unset=True,  # 排除未设置的字段
    response_model_exclude_none=True,   # 排除 None 值
)
async def get_user_clean(user_id: int):
    return user
```

## 3. 文档自动生成

### 3.1 模型文档

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    """
    商品模型

    用于表示系统中的商品信息
    """
    id: int = Field(..., description="商品 ID")
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="商品名称",
        examples=["iPhone 15"],
    )
    price: float = Field(
        ...,
        gt=0,
        description="商品价格（元）",
        examples=[6999.00],
    )
    description: str | None = Field(
        None,
        max_length=500,
        description="商品描述",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "iPhone 15",
                    "price": 6999.00,
                    "description": "最新款苹果手机",
                }
            ]
        }
    }
```

### 3.2 端点文档

```python
@app.post(
    "/items",
    response_model=Item,
    status_code=201,
    summary="创建商品",
    description="创建一个新的商品记录",
    response_description="创建成功返回商品信息",
    tags=["商品管理"],
    responses={
        201: {
            "description": "创建成功",
            "content": {
                "application/json": {
                    "example": {"id": 1, "name": "iPhone", "price": 6999}
                }
            }
        },
        422: {"description": "验证失败"},
    },
)
async def create_item(item: ItemCreate):
    """
    创建商品

    - **name**: 商品名称
    - **price**: 商品价格，必须大于 0
    - **description**: 可选的商品描述
    """
    return item
```

## 4. 高级用法

### 4.1 泛型模型

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""
    items: list[T]
    total: int
    page: int
    page_size: int

@app.get("/users", response_model=PaginatedResponse[UserResponse])
async def list_users(page: int = 1, page_size: int = 10):
    users = get_users(page, page_size)
    return PaginatedResponse(
        items=users,
        total=100,
        page=page,
        page_size=page_size,
    )
```

### 4.2 区分输入输出模型

```python
# 基础模型（共享字段）
class ItemBase(BaseModel):
    name: str
    description: str | None = None
    price: float

# 创建模型（输入）
class ItemCreate(ItemBase):
    pass

# 更新模型（输入，所有字段可选）
class ItemUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    price: float | None = None

# 响应模型（输出）
class ItemResponse(ItemBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}
```

### 4.3 自定义 JSON 编码

```python
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, ConfigDict

class Order(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.strftime("%Y-%m-%d %H:%M:%S"),
            Decimal: lambda v: float(v),
        }
    )

    id: int
    amount: Decimal
    created_at: datetime
```

### 4.4 计算字段

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

@app.post("/rectangles")
async def create_rectangle(rect: Rectangle):
    return rect  # 响应会包含 area 字段
```

## 5. 常见模式

### 5.1 统一响应格式

```python
class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应格式"""
    success: bool = True
    data: T | None = None
    message: str | None = None

@app.get("/users/{user_id}", response_model=ApiResponse[UserResponse])
async def get_user(user_id: int):
    user = db.get_user(user_id)
    return ApiResponse(data=user)
```

### 5.2 请求日志

```python
class RequestLog(BaseModel):
    """请求日志模型"""
    method: str
    path: str
    body: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_log_string(self) -> str:
        return f"[{self.timestamp}] {self.method} {self.path}"
```

## Python vs JavaScript 对比

| 特性 | Pydantic (Python) | Zod (TypeScript) |
|------|-------------------|------------------|
| 定义 | `class Model(BaseModel)` | `z.object({...})` |
| 验证 | 自动 | `.parse()` |
| 类型 | 原生类型注解 | 需要 `.infer<>` |
| 嵌套 | 直接嵌套 | `.extend()` |
| 可选 | `| None` | `.optional()` |
| 默认值 | `= value` | `.default()` |
| 自定义 | `@field_validator` | `.refine()` |

