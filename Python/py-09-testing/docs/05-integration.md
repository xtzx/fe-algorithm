# 05. 集成测试

## 本节目标

- 理解集成测试的作用
- 测试 HTTP 服务
- 数据库测试策略

---

## 测试金字塔

```
         ╱╲
        ╱E2E╲         5% - 端到端测试
       ╱────╲
      ╱ 集成 ╲        20% - 集成测试
     ╱────────╲
    ╱   单元   ╲      75% - 单元测试
   ╱────────────╲
```

---

## 测试 HTTP 服务

### 使用 httpx + respx

```python
import httpx
import respx
import pytest

# 被测代码
async def fetch_user(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()

# 测试
@pytest.mark.asyncio
@respx.mock
async def test_fetch_user():
    # 设置 mock 响应
    respx.get("https://api.example.com/users/1").mock(
        return_value=httpx.Response(
            200,
            json={"id": 1, "name": "Alice"}
        )
    )

    result = await fetch_user(1)

    assert result["name"] == "Alice"

@pytest.mark.asyncio
@respx.mock
async def test_fetch_user_not_found():
    respx.get("https://api.example.com/users/999").mock(
        return_value=httpx.Response(404)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_user(999)
```

### 使用 responses（同步）

```python
import responses
import requests

@responses.activate
def test_sync_api():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"result": "test"},
        status=200
    )

    response = requests.get("https://api.example.com/data")
    assert response.json()["result"] == "test"
```

---

## 测试 FastAPI

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# 应用代码
app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"id": user_id, "name": f"User {user_id}"}

@app.post("/users")
def create_user(name: str):
    return {"id": 1, "name": name}

# 测试
@pytest.fixture
def client():
    return TestClient(app)

def test_get_user(client):
    response = client.get("/users/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1

def test_create_user(client):
    response = client.post("/users", params={"name": "Alice"})
    assert response.status_code == 200
    assert response.json()["name"] == "Alice"
```

### 异步 FastAPI 测试

```python
import pytest
from httpx import AsyncClient

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_async_endpoint(async_client):
    response = await async_client.get("/users/1")
    assert response.status_code == 200
```

---

## 数据库测试

### 策略选择

| 策略 | 说明 | 速度 | 真实性 |
|------|------|------|--------|
| Mock | 完全模拟 | 快 | 低 |
| 内存数据库 | SQLite :memory: | 快 | 中 |
| 测试容器 | Docker 真实数据库 | 慢 | 高 |
| 事务回滚 | 每次测试回滚 | 中 | 高 |

### SQLite 内存数据库

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()

def test_create_user(db_session):
    user = User(name="Alice")
    db_session.add(user)
    db_session.commit()

    assert db_session.query(User).count() == 1
```

### 事务回滚

```python
@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    transaction.rollback()  # 回滚所有更改
    connection.close()
```

### 使用 testcontainers

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture
def db_url(postgres_container):
    return postgres_container.get_connection_url()
```

---

## 测试外部服务

### 使用 fixture mock

```python
# conftest.py
import pytest

@pytest.fixture
def mock_external_api(mocker):
    """Mock 外部 API"""
    mock = mocker.patch("myapp.services.external_api")
    mock.get_data.return_value = {"result": "test"}
    return mock

def test_with_external_api(mock_external_api):
    result = my_service.process()
    mock_external_api.get_data.assert_called_once()
```

### 使用 VCR（录制/回放）

```python
import pytest
import vcr

@vcr.use_cassette("tests/cassettes/api_response.yaml")
def test_external_api():
    # 第一次运行会录制真实响应
    # 之后回放录制的响应
    response = requests.get("https://api.example.com/data")
    assert response.status_code == 200
```

---

## 异步测试

```python
import pytest
import asyncio

# 配置 pytest-asyncio
# pyproject.toml: asyncio_mode = "auto"

async def async_add(a, b):
    await asyncio.sleep(0.1)
    return a + b

@pytest.mark.asyncio
async def test_async_add():
    result = await async_add(1, 2)
    assert result == 3

# 测试并发
@pytest.mark.asyncio
async def test_concurrent():
    results = await asyncio.gather(
        async_add(1, 2),
        async_add(3, 4),
        async_add(5, 6),
    )
    assert results == [3, 7, 11]
```

---

## 集成测试标记

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )

# 测试文件
@pytest.mark.integration
def test_database_connection():
    pass

@pytest.mark.integration
def test_external_api():
    pass
```

```bash
# 只运行集成测试
pytest -m integration

# 排除集成测试
pytest -m "not integration"
```

---

## 本节要点

1. **测试金字塔**：单元 > 集成 > E2E
2. **HTTP 测试**：respx/responses mock 请求
3. **FastAPI**：使用 TestClient
4. **数据库**：内存数据库或事务回滚
5. **异步**：pytest.mark.asyncio
6. **标记**：用 mark 分类测试

