# 07. 练习题

## 练习 1：基本测试

为以下函数编写测试：

```python
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("折扣必须在 0-100 之间")
    return price * (1 - discount_percent / 100)
```

要求：
- 正常情况测试
- 边界值测试（0%, 100%）
- 异常测试

---

## 练习 2：参数化测试

使用 `@pytest.mark.parametrize` 测试以下函数：

```python
def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]
```

测试用例：
- "radar" → True
- "hello" → False
- "A man a plan a canal Panama" → True

---

## 练习 3：Fixture

创建一个 fixture 来提供临时用户数据：

```python
@pytest.fixture
def user():
    # 返回一个用户字典
    pass

def test_user_name(user):
    assert user["name"] is not None
```

---

## 练习 4：Mock HTTP 请求

测试以下函数，mock 掉 requests.get：

```python
import requests

def get_github_user(username: str) -> dict:
    response = requests.get(f"https://api.github.com/users/{username}")
    response.raise_for_status()
    return response.json()
```

---

## 练习 5：Mock 时间

测试一个依赖当前时间的函数：

```python
from datetime import datetime

def is_weekend() -> bool:
    return datetime.now().weekday() >= 5

def get_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "早上好"
    elif hour < 18:
        return "下午好"
    else:
        return "晚上好"
```

---

## 练习 6：测试异步函数

测试以下异步函数：

```python
import asyncio

async def fetch_all(urls: list[str]) -> list[dict]:
    async def fetch_one(url):
        await asyncio.sleep(0.1)  # 模拟网络请求
        return {"url": url, "status": 200}

    return await asyncio.gather(*[fetch_one(url) for url in urls])
```

---

## 练习 7：conftest.py

创建一个 conftest.py，包含：
- session 级别的数据库连接 fixture
- function 级别的测试数据 fixture
- 自定义 marker

---

## 练习 8：覆盖率

1. 运行测试并生成覆盖率报告
2. 识别未覆盖的代码
3. 添加测试达到 90% 覆盖率

```bash
pytest --cov=src --cov-report=html
```

---

## 练习 9：集成测试

为一个 FastAPI 应用编写集成测试：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items")
def create_item(name: str, price: float):
    return {"name": name, "price": price}
```

---

## 练习 10：测试工厂

创建一个 User 工厂 fixture：

```python
@pytest.fixture
def make_user():
    """工厂 fixture，可创建多个用户"""
    def _make_user(name="Default", age=25):
        return {"name": name, "age": age}
    return _make_user

def test_multiple_users(make_user):
    user1 = make_user("Alice", 30)
    user2 = make_user("Bob", 25)
    # 测试...
```

---

## 练习 11：使用 monkeypatch

使用 monkeypatch 测试环境变量依赖：

```python
import os

def get_api_key() -> str:
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError("API_KEY not set")
    return key
```

---

## 练习 12：测试数据库操作

为以下代码编写测试（使用 SQLite 内存数据库）：

```python
class UserRepository:
    def __init__(self, db):
        self.db = db

    def create(self, name: str, email: str) -> dict:
        # 创建用户
        pass

    def get_by_id(self, user_id: int) -> dict | None:
        # 获取用户
        pass
```

---

## 练习 13：并行测试

1. 安装 pytest-xdist
2. 运行并行测试
3. 确保测试之间没有依赖

```bash
pip install pytest-xdist
pytest -n auto
```

---

## 练习 14：测试标记

创建标记并按标记运行测试：

```python
@pytest.mark.slow
def test_slow():
    import time
    time.sleep(2)

@pytest.mark.unit
def test_fast():
    assert True
```

```bash
pytest -m "not slow"
```

---

## 练习 15：测试私有方法

讨论：如何测试私有方法？应该测试吗？

```python
class Calculator:
    def add(self, a, b):
        return self._validate(a) + self._validate(b)

    def _validate(self, n):
        if not isinstance(n, (int, float)):
            raise TypeError("Must be a number")
        return n
```

