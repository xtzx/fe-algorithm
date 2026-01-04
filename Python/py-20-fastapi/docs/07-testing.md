# 测试

## 概述

FastAPI 提供完整的测试支持：

1. **TestClient** - 同步测试
2. **异步测试** - pytest-asyncio
3. **依赖覆盖** - Mock 依赖
4. **fixture** - 测试数据

## 1. TestClient

### 1.1 基础测试

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert "id" in response.json()

def test_create_item():
    response = client.post(
        "/items",
        json={"name": "Test Item", "price": 10.0},
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Item"
```

### 1.2 带认证的测试

```python
def test_protected_route():
    # 先登录获取 token
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "testpass"},
    )
    token = response.json()["access_token"]

    # 使用 token 访问保护的路由
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
```

### 1.3 测试错误情况

```python
def test_item_not_found():
    response = client.get("/items/99999")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"

def test_validation_error():
    response = client.post(
        "/items",
        json={"name": "", "price": -10},  # 无效数据
    )
    assert response.status_code == 422
    assert "details" in response.json()
```

## 2. 异步测试

### 2.1 配置 pytest-asyncio

```python
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### 2.2 异步测试示例

```python
import pytest
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_async_root():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_async_create_item():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/items",
            json={"name": "Async Item", "price": 20.0},
        )
        assert response.status_code == 201
```

### 2.3 异步 fixture

```python
import pytest

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_with_fixture(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
```

## 3. 依赖覆盖

### 3.1 覆盖数据库依赖

```python
from api.main import app
from api.dependencies import get_db

# 测试用数据库
def override_get_db():
    db = TestDatabase()
    try:
        yield db
    finally:
        db.close()

def test_with_test_db():
    # 覆盖依赖
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    response = client.get("/items")

    assert response.status_code == 200

    # 清理
    app.dependency_overrides.clear()
```

### 3.2 覆盖认证依赖

```python
from api.dependencies.auth import get_current_user
from api.schemas.user import User

# 模拟用户
def override_get_current_user():
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        is_active=True,
        scopes=["user"],
    )

def test_protected_route():
    app.dependency_overrides[get_current_user] = override_get_current_user

    client = TestClient(app)
    response = client.get("/users/me")

    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

    app.dependency_overrides.clear()
```

### 3.3 覆盖为异步依赖

```python
async def override_get_async_resource():
    return MockResource()

@pytest.mark.asyncio
async def test_with_async_override():
    app.dependency_overrides[get_async_resource] = override_get_async_resource

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/resource")
        assert response.status_code == 200

    app.dependency_overrides.clear()
```

## 4. Fixture

### 4.1 conftest.py

```python
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from api.main import app
from api.dependencies import get_db
from api.dependencies.auth import get_current_user

@pytest.fixture
def client():
    """同步测试客户端"""
    with TestClient(app) as c:
        yield c

@pytest.fixture
async def async_client():
    """异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.fixture
def test_user():
    """测试用户"""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        is_active=True,
        scopes=["user"],
    )

@pytest.fixture
def admin_user():
    """管理员用户"""
    return User(
        id=2,
        username="admin",
        email="admin@example.com",
        is_active=True,
        scopes=["user", "admin"],
    )

@pytest.fixture
def authenticated_client(client, test_user):
    """已认证的客户端"""
    app.dependency_overrides[get_current_user] = lambda: test_user
    yield client
    app.dependency_overrides.clear()

@pytest.fixture
def admin_client(client, admin_user):
    """管理员客户端"""
    app.dependency_overrides[get_current_user] = lambda: admin_user
    yield client
    app.dependency_overrides.clear()
```

### 4.2 使用 fixture

```python
def test_list_items(client):
    response = client.get("/items")
    assert response.status_code == 200

def test_get_current_user(authenticated_client):
    response = authenticated_client.get("/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_admin_delete_user(admin_client):
    response = admin_client.delete("/users/1")
    assert response.status_code == 204
```

## 5. Mock 外部服务

### 5.1 使用 unittest.mock

```python
from unittest.mock import patch, MagicMock

def test_external_api_call():
    # Mock 外部 HTTP 调用
    with patch("api.services.external.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "mocked"},
        )

        client = TestClient(app)
        response = client.get("/external-data")

        assert response.status_code == 200
        mock_get.assert_called_once()
```

### 5.2 使用 respx (异步 HTTP Mock)

```python
import respx
import httpx

@pytest.mark.asyncio
@respx.mock
async def test_async_external_call():
    # Mock 异步 HTTP 调用
    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, json={"result": "mocked"})
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/external-data")
        assert response.status_code == 200
```

## 6. 数据库测试

### 6.1 使用测试数据库

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 测试数据库配置
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine)

@pytest.fixture
def db():
    """创建测试数据库会话"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db):
    """使用测试数据库的客户端"""
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
```

### 6.2 事务回滚

```python
@pytest.fixture
def db():
    """使用事务回滚而不是删除重建"""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
```

## 7. 测试组织

### 7.1 目录结构

```
tests/
├── __init__.py
├── conftest.py          # 共享 fixture
├── test_auth.py         # 认证测试
├── test_users.py        # 用户路由测试
├── test_items.py        # 商品路由测试
├── test_services/       # 服务层测试
│   ├── test_user_service.py
│   └── test_item_service.py
└── factories/           # 测试数据工厂
    └── user_factory.py
```

### 7.2 测试分类

```python
import pytest

@pytest.mark.unit
def test_password_hash():
    """单元测试"""
    hashed = get_password_hash("password")
    assert verify_password("password", hashed)

@pytest.mark.integration
def test_create_user_flow(client):
    """集成测试"""
    response = client.post("/users", json={"username": "test", ...})
    assert response.status_code == 201

@pytest.mark.e2e
def test_user_registration_login(client):
    """端到端测试"""
    # 注册
    client.post("/register", json={...})
    # 登录
    response = client.post("/token", data={...})
    # 访问保护资源
    client.get("/users/me", headers={...})
```

### 7.3 运行测试

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_auth.py

# 运行特定测试
pytest tests/test_auth.py::test_login

# 按标记运行
pytest -m unit
pytest -m integration

# 带覆盖率
pytest --cov=api --cov-report=html

# 并行运行
pytest -n auto
```

## Python vs JavaScript 对比

| 特性 | FastAPI + pytest | Express + Jest |
|------|-----------------|----------------|
| 客户端 | `TestClient` | `supertest` |
| 异步 | `pytest-asyncio` | 内置 |
| Mock | `unittest.mock` | `jest.mock()` |
| 覆盖 | `dependency_overrides` | DI 容器 |
| fixture | `@pytest.fixture` | `beforeEach` |

