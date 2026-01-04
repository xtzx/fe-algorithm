# 面试高频题

## 1. FastAPI 和 Flask 的区别？

### 答案

| 特性 | FastAPI | Flask |
|------|---------|-------|
| **类型** | ASGI (异步) | WSGI (同步) |
| **性能** | 高（异步 I/O） | 中等 |
| **类型提示** | 内置，必须 | 可选 |
| **验证** | 自动（Pydantic） | 手动（WTForms/Marshmallow） |
| **文档** | 自动生成 | 手动（Flask-Swagger） |
| **异步支持** | 原生 | Flask 2.0+ 支持 |
| **学习曲线** | 稍陡（需了解类型） | 平缓 |

**适用场景**：
- **FastAPI** - 高性能 API、微服务、需要自动文档
- **Flask** - 简单应用、需要大量扩展、团队熟悉度高

```python
# Flask
@app.route("/items/<int:item_id>")
def get_item(item_id):
    return jsonify({"item_id": item_id})

# FastAPI
@app.get("/items/{item_id}")
async def get_item(item_id: int):  # 类型自动验证
    return {"item_id": item_id}
```

---

## 2. 依赖注入是什么？

### 答案

**依赖注入（DI）** 是一种设计模式，将组件的依赖项从外部传入，而不是在组件内部创建。

**FastAPI 中的应用**：

```python
# 1. 共享逻辑
async def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

# 2. 注入到路由
@app.get("/users")
async def list_users(db: Database = Depends(get_db)):
    return db.get_all_users()
```

**优点**：
1. **解耦** - 路由不关心数据库如何创建
2. **可测试** - 易于 Mock
3. **复用** - 多个路由共享同一依赖
4. **清理** - 自动管理资源生命周期

**依赖链**：
```python
def get_db(): ...
def get_repo(db = Depends(get_db)): return UserRepository(db)
def get_service(repo = Depends(get_repo)): return UserService(repo)

@app.get("/users")
async def list_users(service = Depends(get_service)):
    return service.get_all()
```

---

## 3. 如何处理跨域？

### 答案

使用 `CORSMiddleware`：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=600,  # 预检请求缓存
)
```

**关键参数**：
- `allow_origins` - 允许的源（生产环境不要用 `*`）
- `allow_credentials` - 允许 Cookie
- `allow_methods` - 允许的 HTTP 方法
- `allow_headers` - 允许的请求头
- `expose_headers` - 暴露给前端的响应头
- `max_age` - 预检请求缓存时间

---

## 4. 如何实现 JWT 认证？

### 答案

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone

# 1. 配置
SECRET_KEY = "secret"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 2. 创建 Token
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 3. 验证 Token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)

    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401)
    return user

# 4. 使用
@app.get("/users/me")
async def read_me(user = Depends(get_current_user)):
    return user
```

---

## 5. 如何测试 FastAPI 应用？

### 答案

**1. 同步测试（TestClient）**：
```python
from fastapi.testclient import TestClient

def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
```

**2. 异步测试（httpx）**：
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
```

**3. 覆盖依赖**：
```python
def test_with_mock_db():
    app.dependency_overrides[get_db] = lambda: MockDB()
    client = TestClient(app)
    response = client.get("/items")
    app.dependency_overrides.clear()
```

---

## 6. 如何处理文件上传？

### 答案

```python
from fastapi import UploadFile, File, HTTPException

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "只允许上传图片")

    # 验证文件大小
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(400, "文件过大")

    # 保存文件
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(contents)

    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type,
    }

# 多文件上传
@app.post("/uploads")
async def upload_files(files: list[UploadFile] = File(...)):
    return [{"filename": f.filename} for f in files]
```

---

## 7. 如何实现分页？

### 答案

```python
from pydantic import BaseModel, Field
from typing import Generic, TypeVar

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int

class Pagination:
    def __init__(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
    ):
        self.page = page
        self.page_size = page_size
        self.skip = (page - 1) * page_size

@app.get("/items", response_model=PaginatedResponse[Item])
async def list_items(pagination: Pagination = Depends()):
    total = db.count_items()
    items = db.get_items(skip=pagination.skip, limit=pagination.page_size)

    return PaginatedResponse(
        items=items,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        pages=(total + pagination.page_size - 1) // pagination.page_size,
    )
```

---

## 8. 如何处理后台任务？

### 答案

```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    """后台发送邮件"""
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks,
):
    # 添加后台任务
    background_tasks.add_task(send_email, email, "Hello!")

    # 立即返回
    return {"message": "通知正在发送"}

# 依赖中添加后台任务
async def log_request(
    background_tasks: BackgroundTasks,
    request: Request,
):
    def write_log(path: str):
        with open("log.txt", "a") as f:
            f.write(f"{path}\n")

    background_tasks.add_task(write_log, request.url.path)
```

**复杂任务用 Celery**：
```python
from celery import Celery

celery = Celery("tasks", broker="redis://localhost")

@celery.task
def heavy_computation(data):
    # 耗时操作
    return result

@app.post("/process")
async def process_data(data: dict):
    task = heavy_computation.delay(data)
    return {"task_id": task.id}
```

---

## 9. 如何实现 WebSocket？

### 答案

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)

    async def broadcast(self, message: str):
        for conn in self.connections:
            await conn.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    await manager.broadcast(f"{client_id} 加入了聊天")

    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"{client_id} 离开了聊天")
```

---

## 10. FastAPI 的性能为什么好？

### 答案

**1. 异步 I/O（ASGI）**：
- 基于 `asyncio`，非阻塞 I/O
- 单线程处理大量并发连接
- 与 NodeJS 事件循环类似

**2. Starlette 底层**：
- 高性能 ASGI 框架
- 原生支持 WebSocket、HTTP/2

**3. Pydantic v2**：
- Rust 编写的核心
- 比 v1 快 5-50 倍

**4. 编译优化**：
- 路由在启动时编译
- 依赖图预计算

**5. 最小开销**：
- 中间件轻量
- 没有复杂的 ORM 抽象

**基准测试对比**：
```
FastAPI:    ~15,000 req/s
Flask:      ~2,000 req/s
Django:     ~1,500 req/s
Express:    ~10,000 req/s
```

---

## 附加题

### 11. 如何处理大文件流式响应？

```python
from fastapi.responses import StreamingResponse

def generate_large_file():
    for i in range(1000000):
        yield f"line {i}\n"

@app.get("/download")
async def download():
    return StreamingResponse(
        generate_large_file(),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=large.txt"},
    )
```

### 12. 如何实现限流？

```python
from fastapi import Request
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: dict[str, list] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []

        # 清理过期请求
        self.requests[client_id] = [
            t for t in self.requests[client_id] if now - t < self.window
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})
    return await call_next(request)
```

