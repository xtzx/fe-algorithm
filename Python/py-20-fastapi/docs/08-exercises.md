# 练习题

## 基础练习

### 练习 1: Hello World API

创建一个简单的 FastAPI 应用：

```python
# 要求:
# 1. GET / 返回 {"message": "Hello World"}
# 2. GET /items/{item_id} 返回 {"item_id": item_id}
# 3. GET /items 支持 skip 和 limit 查询参数

# 你的代码:
from fastapi import FastAPI

app = FastAPI()

# TODO: 实现路由
```

### 练习 2: 请求体验证

使用 Pydantic 创建商品创建端点：

```python
# 要求:
# 1. POST /items 接收 ItemCreate 请求体
# 2. name: 1-100 字符
# 3. price: 必须大于 0
# 4. quantity: 默认为 0，必须 >= 0
# 5. 返回创建的商品（包含 id）

from pydantic import BaseModel, Field

class ItemCreate(BaseModel):
    # TODO: 定义字段
    pass

@app.post("/items")
async def create_item(item: ItemCreate):
    # TODO: 实现
    pass
```

### 练习 3: 查询参数过滤

实现带过滤功能的商品列表：

```python
# 要求:
# 1. GET /items 支持以下查询参数:
#    - skip: 跳过记录数（默认 0）
#    - limit: 最大返回数（默认 10，最大 100）
#    - q: 搜索关键词（可选）
#    - min_price: 最低价格（可选）
#    - max_price: 最高价格（可选）
#    - category: 分类（可选）

@app.get("/items")
async def list_items(
    # TODO: 定义参数
):
    # TODO: 实现过滤逻辑
    pass
```

### 练习 4: 响应模型

创建不同场景的响应模型：

```python
# 要求:
# 1. 创建 UserBase, UserCreate, User, UserInDB 模型
# 2. UserInDB 包含 hashed_password，但 User 不包含
# 3. GET /users/{user_id} 返回 User（不暴露密码）
# 4. POST /users 接收 UserCreate，返回 User

# TODO: 定义模型和路由
```

### 练习 5: 状态码

正确使用 HTTP 状态码：

```python
# 要求:
# 1. POST /items - 创建成功返回 201
# 2. PUT /items/{id} - 更新成功返回 200
# 3. DELETE /items/{id} - 删除成功返回 204
# 4. GET /items/{id} - 不存在返回 404
# 5. POST /items - 验证失败返回 422

# TODO: 实现路由
```

## 进阶练习

### 练习 6: 依赖注入

实现分页依赖：

```python
# 要求:
# 1. 创建 Pagination 依赖类
# 2. 支持 page（默认 1）和 page_size（默认 10，最大 100）
# 3. 计算 skip 和 limit
# 4. 在多个路由中复用

class Pagination:
    def __init__(self, page: int = 1, page_size: int = 10):
        # TODO: 实现
        pass

@app.get("/items")
async def list_items(pagination: Pagination = Depends()):
    # TODO: 使用 pagination
    pass
```

### 练习 7: 数据库依赖

实现带资源清理的数据库依赖：

```python
# 要求:
# 1. 创建 get_db 依赖函数
# 2. 使用 yield 提供数据库会话
# 3. 确保请求结束后关闭连接
# 4. 处理异常时也要关闭连接

def get_db():
    # TODO: 实现
    pass
```

### 练习 8: JWT 认证

实现完整的 JWT 认证系统：

```python
# 要求:
# 1. POST /register - 用户注册
# 2. POST /token - 登录获取 token
# 3. GET /users/me - 获取当前用户（需认证）
# 4. 密码使用 bcrypt 加密
# 5. Token 有效期 30 分钟

# TODO: 实现认证系统
```

### 练习 9: 权限控制

实现基于角色的权限控制：

```python
# 要求:
# 1. 用户有 role 字段（user, admin）
# 2. GET /users - 所有登录用户可访问
# 3. DELETE /users/{id} - 只有 admin 可访问
# 4. PUT /users/{id} - admin 或本人可访问

def require_role(roles: list[str]):
    # TODO: 实现
    pass
```

### 练习 10: 中间件

实现请求日志中间件：

```python
# 要求:
# 1. 记录请求方法和路径
# 2. 记录响应状态码
# 3. 记录处理时间
# 4. 生成并返回 X-Request-ID 响应头

class LoggingMiddleware(BaseHTTPMiddleware):
    # TODO: 实现
    pass
```

### 练习 11: 错误处理

实现统一错误处理：

```python
# 要求:
# 1. 创建 AppException 基类
# 2. 创建 NotFoundException, ValidationException 等子类
# 3. 注册异常处理器
# 4. 统一错误响应格式: {"success": False, "error_code": "...", "message": "..."}

# TODO: 实现异常类和处理器
```

### 练习 12: 文件上传

实现文件上传功能：

```python
# 要求:
# 1. POST /upload - 单文件上传
# 2. POST /uploads - 多文件上传
# 3. 限制文件大小（最大 5MB）
# 4. 只允许图片格式（jpg, png, gif）
# 5. 返回文件信息（名称、大小、类型）

from fastapi import UploadFile, File

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # TODO: 实现
    pass
```

### 练习 13: 后台任务

实现后台任务：

```python
# 要求:
# 1. POST /send-email - 发送邮件（后台执行）
# 2. 立即返回 {"message": "邮件正在发送"}
# 3. 后台任务记录发送日志
# 4. 支持发送给多个收件人

from fastapi import BackgroundTasks

@app.post("/send-email")
async def send_email(
    email: str,
    background_tasks: BackgroundTasks,
):
    # TODO: 实现
    pass
```

### 练习 14: WebSocket

实现简单的聊天功能：

```python
# 要求:
# 1. WebSocket /ws/{client_id}
# 2. 连接时广播 "用户 xxx 加入"
# 3. 收到消息时广播给所有用户
# 4. 断开时广播 "用户 xxx 离开"

from fastapi import WebSocket

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # TODO: 实现
    pass
```

### 练习 15: 完整 CRUD

实现完整的商品管理 API：

```python
# 要求:
# 1. GET /items - 列表（分页、搜索、过滤）
# 2. GET /items/{id} - 详情
# 3. POST /items - 创建（需认证）
# 4. PUT /items/{id} - 更新（所有者或管理员）
# 5. DELETE /items/{id} - 删除（所有者或管理员）
# 6. 完整的错误处理
# 7. 完整的响应模型
# 8. 完整的测试

# TODO: 实现完整功能
```

## 参考答案

### 练习 1 答案

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

@app.get("/items")
async def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    return {"skip": skip, "limit": limit}
```

### 练习 6 答案

```python
from fastapi import Query

class Pagination:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    ):
        self.page = page
        self.page_size = page_size
        self.skip = (page - 1) * page_size
        self.limit = page_size

@app.get("/items")
async def list_items(pagination: Pagination = Depends()):
    items = db.get_items(skip=pagination.skip, limit=pagination.limit)
    return {
        "items": items,
        "page": pagination.page,
        "page_size": pagination.page_size,
    }
```

### 练习 10 答案

```python
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        print(f"[{request_id}] --> {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = (time.perf_counter() - start_time) * 1000
        print(
            f"[{request_id}] <-- {request.method} {request.url.path} "
            f"{response.status_code} {process_time:.2f}ms"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        return response
```

