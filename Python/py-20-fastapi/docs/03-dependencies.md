# 依赖注入

## 概述

依赖注入（Dependency Injection）是 FastAPI 的核心特性，用于：

1. **共享逻辑** - 数据库连接、认证
2. **代码复用** - 避免重复代码
3. **解耦** - 便于测试和替换

## 1. 基础用法

### 1.1 函数依赖

```python
from fastapi import Depends

# 简单的依赖函数
def common_parameters(
    skip: int = 0,
    limit: int = 100,
):
    return {"skip": skip, "limit": limit}

@app.get("/items")
async def list_items(params: dict = Depends(common_parameters)):
    return {"params": params}

@app.get("/users")
async def list_users(params: dict = Depends(common_parameters)):
    return {"params": params}
```

### 1.2 类依赖

```python
class CommonQueryParams:
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        q: str | None = None,
    ):
        self.skip = skip
        self.limit = limit
        self.q = q

@app.get("/items")
async def list_items(params: CommonQueryParams = Depends()):
    return {
        "skip": params.skip,
        "limit": params.limit,
        "q": params.q,
    }
```

### 1.3 异步依赖

```python
async def get_async_resource():
    resource = await fetch_resource()
    return resource

@app.get("/items")
async def list_items(resource = Depends(get_async_resource)):
    return {"resource": resource}
```

## 2. 依赖的依赖

```python
# 底层依赖
def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

# 中间层依赖
def get_user_repository(db: Database = Depends(get_db)):
    return UserRepository(db)

# 顶层依赖
def get_user_service(repo: UserRepository = Depends(get_user_repository)):
    return UserService(repo)

@app.get("/users")
async def list_users(service: UserService = Depends(get_user_service)):
    return service.get_all()
```

## 3. 数据库连接

### 3.1 同步数据库

```python
from contextlib import contextmanager
from typing import Generator

def get_db() -> Generator:
    """数据库连接依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

### 3.2 异步数据库

```python
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """异步数据库连接依赖"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@app.get("/users")
async def list_users(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

## 4. 认证依赖

### 4.1 简单认证

```python
from fastapi import Header, HTTPException

async def verify_token(x_token: str = Header(...)):
    """验证 API Token"""
    if x_token != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return x_token

@app.get("/protected")
async def protected_route(token: str = Depends(verify_token)):
    return {"token": token}
```

### 4.2 OAuth2 认证

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

### 4.3 权限检查

```python
def require_role(required_roles: list[str]):
    """创建角色检查依赖"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in required_roles:
            raise HTTPException(status_code=403, detail="权限不足")
        return current_user
    return role_checker

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_role(["admin"])),
):
    return {"deleted": user_id}
```

## 5. 路由级依赖

### 5.1 应用全局依赖

```python
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "valid-key":
        raise HTTPException(status_code=403)
    return x_api_key

# 全局依赖
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

### 5.2 路由器级依赖

```python
router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_role(["admin"]))],
)

@router.get("/users")
async def list_admin_users():
    # 所有 /admin/* 路由都需要 admin 权限
    return []
```

### 5.3 单路由依赖

```python
@app.get(
    "/protected",
    dependencies=[Depends(verify_token), Depends(rate_limit)],
)
async def protected_route():
    return {"message": "protected"}
```

## 6. 依赖覆盖（测试）

```python
# 原始依赖
def get_db():
    return ProductionDatabase()

# 测试中覆盖
def override_get_db():
    return TestDatabase()

# 在测试中使用
app.dependency_overrides[get_db] = override_get_db

# 测试完成后清理
app.dependency_overrides.clear()
```

### 完整测试示例

```python
from fastapi.testclient import TestClient

def test_list_items():
    # 覆盖数据库依赖
    def override_db():
        return MockDatabase()

    app.dependency_overrides[get_db] = override_db

    client = TestClient(app)
    response = client.get("/items")

    assert response.status_code == 200

    # 清理
    app.dependency_overrides.clear()
```

## 7. 高级用法

### 7.1 带参数的依赖

```python
def pagination(max_limit: int = 100):
    """创建分页依赖工厂"""
    def paginate(
        page: int = 1,
        limit: int = 10,
    ):
        if limit > max_limit:
            limit = max_limit
        skip = (page - 1) * limit
        return {"skip": skip, "limit": limit}
    return paginate

# 使用
@app.get("/items")
async def list_items(
    pagination: dict = Depends(pagination(max_limit=50)),
):
    return pagination
```

### 7.2 缓存依赖

```python
from functools import lru_cache

@lru_cache()
def get_settings():
    """缓存配置（只加载一次）"""
    return Settings()

@app.get("/info")
async def info(settings: Settings = Depends(get_settings)):
    return {"app_name": settings.app_name}
```

### 7.3 上下文依赖

```python
from contextvars import ContextVar

# 请求上下文
request_id_var: ContextVar[str] = ContextVar("request_id")

async def set_request_context():
    """设置请求上下文"""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id

def get_request_id() -> str:
    """获取当前请求 ID"""
    return request_id_var.get()
```

## 8. 依赖执行顺序

```python
async def dep_a():
    print("A start")
    yield "A"
    print("A end")

async def dep_b(a: str = Depends(dep_a)):
    print("B start")
    yield f"B({a})"
    print("B end")

@app.get("/")
async def root(b: str = Depends(dep_b)):
    print("Handler")
    return {"value": b}

# 执行顺序:
# A start
# B start
# Handler
# B end
# A end
```

## Python vs JavaScript 对比

| 特性 | FastAPI | NestJS |
|------|---------|--------|
| 声明 | `Depends()` | `@Injectable()` |
| 注入 | 函数参数 | 构造函数 |
| 生命周期 | yield (请求范围) | Scope |
| 覆盖 | `dependency_overrides` | `overrideProvider()` |
| 全局 | `FastAPI(dependencies=[])` | `@Global()` |

