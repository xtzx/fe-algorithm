# FastAPI 基础

## 概述

FastAPI 是一个现代、快速的 Python Web 框架，基于标准 Python 类型提示。

### 核心特性

1. **高性能** - 与 NodeJS 和 Go 相当
2. **类型安全** - 基于 Pydantic 和类型提示
3. **自动文档** - Swagger UI 和 ReDoc
4. **异步支持** - 原生 async/await

## 1. 创建应用

```python
from fastapi import FastAPI

# 创建应用实例
app = FastAPI(
    title="My API",
    description="API 描述",
    version="1.0.0",
)

# 根路由
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### 运行应用

```bash
# 开发模式（热重载）
uvicorn main:app --reload

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 2. 路由与请求处理

### 2.1 HTTP 方法装饰器

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items")
async def list_items():
    """GET 请求"""
    return []

@app.post("/items")
async def create_item():
    """POST 请求"""
    return {"created": True}

@app.put("/items/{item_id}")
async def update_item(item_id: int):
    """PUT 请求"""
    return {"updated": item_id}

@app.patch("/items/{item_id}")
async def partial_update_item(item_id: int):
    """PATCH 请求"""
    return {"patched": item_id}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """DELETE 请求"""
    return {"deleted": item_id}
```

### 2.2 路由分组（Router）

```python
from fastapi import APIRouter

# 创建路由器
router = APIRouter(
    prefix="/users",
    tags=["用户"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def list_users():
    return []

@router.get("/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# 在主应用中注册
app.include_router(router)
```

## 3. 请求参数

### 3.1 路径参数（Path Parameters）

```python
from fastapi import Path

@app.get("/items/{item_id}")
async def read_item(
    item_id: int = Path(
        ...,  # 必填
        title="Item ID",
        description="要获取的商品 ID",
        ge=1,  # >= 1
        le=1000,  # <= 1000
    )
):
    return {"item_id": item_id}
```

### 3.2 查询参数（Query Parameters）

```python
from fastapi import Query

@app.get("/items")
async def list_items(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(10, ge=1, le=100, description="返回的最大记录数"),
    q: str | None = Query(None, min_length=1, max_length=50),
):
    return {"skip": skip, "limit": limit, "q": q}
```

### 3.3 请求体（Request Body）

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    quantity: int = 0

@app.post("/items")
async def create_item(item: Item):
    return item
```

### 3.4 请求头（Headers）

```python
from fastapi import Header

@app.get("/items")
async def read_items(
    x_token: str = Header(..., description="认证令牌"),
    user_agent: str | None = Header(None),
):
    return {"x_token": x_token, "user_agent": user_agent}
```

### 3.5 Cookie

```python
from fastapi import Cookie

@app.get("/items")
async def read_items(
    session_id: str | None = Cookie(None),
):
    return {"session_id": session_id}
```

## 4. 响应模型

### 4.1 指定响应模型

```python
from pydantic import BaseModel

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    # 即使返回更多字段，也只会输出 ItemResponse 中定义的
    return {
        "id": item_id,
        "name": "Test",
        "price": 10.0,
        "secret": "hidden",  # 不会出现在响应中
    }
```

### 4.2 排除未设置的字段

```python
@app.get(
    "/items/{item_id}",
    response_model=ItemResponse,
    response_model_exclude_unset=True,  # 排除未设置的字段
)
async def get_item(item_id: int):
    return {"id": item_id, "name": "Test"}  # price 不会出现
```

### 4.3 多响应模型

```python
from typing import Union

class Cat(BaseModel):
    name: str
    meow: str

class Dog(BaseModel):
    name: str
    bark: str

@app.get("/animals/{animal_id}", response_model=Union[Cat, Dog])
async def get_animal(animal_id: int):
    if animal_id % 2 == 0:
        return Cat(name="Kitty", meow="喵")
    return Dog(name="Buddy", bark="汪")
```

## 5. 状态码

### 5.1 指定状态码

```python
from fastapi import status

@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    return None
```

### 5.2 常用状态码

| 状态码 | 常量 | 含义 |
|--------|------|------|
| 200 | HTTP_200_OK | 成功 |
| 201 | HTTP_201_CREATED | 已创建 |
| 204 | HTTP_204_NO_CONTENT | 无内容 |
| 400 | HTTP_400_BAD_REQUEST | 错误请求 |
| 401 | HTTP_401_UNAUTHORIZED | 未授权 |
| 403 | HTTP_403_FORBIDDEN | 禁止访问 |
| 404 | HTTP_404_NOT_FOUND | 不存在 |
| 422 | HTTP_422_UNPROCESSABLE_ENTITY | 验证失败 |
| 500 | HTTP_500_INTERNAL_SERVER_ERROR | 服务器错误 |

## 6. 同步 vs 异步

```python
# 异步函数（推荐用于 I/O 操作）
@app.get("/async")
async def async_endpoint():
    await some_async_operation()
    return {"type": "async"}

# 同步函数（FastAPI 会在线程池中运行）
@app.get("/sync")
def sync_endpoint():
    some_blocking_operation()
    return {"type": "sync"}
```

### 何时使用异步？

- **使用 async** - 调用异步库（httpx, aiofiles）、异步数据库驱动
- **使用同步** - CPU 密集操作、调用同步库

## 7. 生命周期管理

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("Starting up...")
    await init_database()
    yield
    # 关闭时执行
    print("Shutting down...")
    await close_database()

app = FastAPI(lifespan=lifespan)
```

## 8. 自动文档

FastAPI 自动生成两种交互式文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### 文档增强

```python
@app.get(
    "/items/{item_id}",
    summary="获取商品",
    description="根据 ID 获取商品详情",
    response_description="商品信息",
    responses={
        200: {"description": "成功返回商品信息"},
        404: {"description": "商品不存在"},
    },
)
async def get_item(item_id: int):
    """
    获取指定 ID 的商品:

    - **item_id**: 商品 ID

    返回商品的详细信息
    """
    return {"item_id": item_id}
```

## Python vs JavaScript 对比

| 概念 | FastAPI | Express.js |
|------|---------|------------|
| 路由 | `@app.get("/")` | `app.get("/", ...)` |
| 路径参数 | `{item_id}` + 类型注解 | `:itemId` + `req.params` |
| 查询参数 | `Query()` + 类型注解 | `req.query` |
| 请求体 | Pydantic Model | `req.body` + 手动验证 |
| 验证 | 自动 (Pydantic) | 手动 (Joi/Zod) |
| 文档 | 自动生成 | 手动配置 Swagger |

