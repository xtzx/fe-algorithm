# 08. 面试题

## 1. pytest 和 unittest 的区别？

**答案**：

| 特性 | pytest | unittest |
|------|--------|----------|
| 断言 | `assert x == y` | `self.assertEqual(x, y)` |
| 测试发现 | 自动 | 需继承 TestCase |
| Fixture | 灵活的 fixture 系统 | setUp/tearDown |
| 参数化 | 内置 @parametrize | 需要第三方库 |
| 插件 | 丰富生态 | 有限 |
| 输出 | 友好、彩色 | 基础 |

**推荐**：新项目使用 pytest。

---

## 2. fixture 的 scope 有哪些？

**答案**：

| Scope | 说明 | 生命周期 |
|-------|------|----------|
| function | 每个测试函数 | 默认，最隔离 |
| class | 每个测试类 | 类内共享 |
| module | 每个测试文件 | 文件内共享 |
| session | 整个测试会话 | 所有测试共享 |

```python
@pytest.fixture(scope="session")
def database_connection():
    conn = create_connection()
    yield conn
    conn.close()
```

**选择原则**：
- 需要隔离 → function
- 昂贵资源 → session
- 共享状态小心 → 可能导致测试依赖

---

## 3. 如何 mock 第三方 API？

**答案**：

```python
from unittest.mock import patch

# 方法 1：patch 装饰器
@patch("mymodule.requests.get")
def test_api(mock_get):
    mock_get.return_value.json.return_value = {"data": "test"}
    result = mymodule.fetch_data()
    assert result == {"data": "test"}

# 方法 2：上下文管理器
def test_api():
    with patch("mymodule.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": "test"}
        # 测试代码

# 方法 3：使用 respx（异步）
@respx.mock
async def test_async_api():
    respx.get("https://api.example.com").respond(json={"data": "test"})
```

**关键**：patch 的路径是**使用处**而非定义处。

---

## 4. 如何测试异步函数？

**答案**：

```python
import pytest

# 方法 1：pytest.mark.asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_add(1, 2)
    assert result == 3

# 方法 2：使用 AsyncMock
from unittest.mock import AsyncMock

async def test_with_async_mock():
    mock = AsyncMock(return_value=42)
    result = await mock()
    assert result == 42

# 配置 pyproject.toml
# asyncio_mode = "auto"
```

---

## 5. 如何测试私有方法？

**答案**：

**观点一：不直接测试**
- 私有方法是实现细节
- 通过公共方法间接测试
- 保持测试与实现解耦

**观点二：必要时可以测试**
- 复杂的私有方法可以直接测试
- Python 的"私有"只是约定

```python
class Calculator:
    def add(self, a, b):
        return self._validate(a) + self._validate(b)

    def _validate(self, n):
        # 复杂验证逻辑
        pass

# 推荐：通过 add 测试 _validate
def test_add_with_invalid_input():
    calc = Calculator()
    with pytest.raises(TypeError):
        calc.add("a", 1)

# 或者直接测试（如果必要）
def test_validate_directly():
    calc = Calculator()
    assert calc._validate(5) == 5
```

---

## 6. 什么时候应该 mock，什么时候不应该？

**答案**：

### 应该 Mock
- 外部服务（HTTP API、数据库）
- 文件系统操作
- 时间相关（`datetime.now()`）
- 随机数
- 昂贵的操作

### 不应该 Mock
- 被测代码本身（失去意义）
- 简单的内存对象
- 纯函数

### 原则
```
Mock 在边界：你的代码 <-> 外部系统
```

过度 mock 是设计问题的信号。

---

## 7. 覆盖率 100% 是好事吗？

**答案**：

**不一定**。

**问题**：
1. **边际效益递减**：最后 10% 成本很高
2. **虚假安全感**：覆盖 ≠ 正确
3. **测试质量下降**：为覆盖率写无意义测试
4. **维护成本高**：太多测试难维护

**正确态度**：
- 80% 是合理目标
- 关注测试质量而非数量
- 覆盖关键路径
- 低覆盖率是问题信号

```
高覆盖率 ≠ 高质量
低覆盖率 = 可能有问题
```

---

## 8. 如何测试异常？

**答案**：

```python
import pytest

# 方法 1：pytest.raises
def test_exception():
    with pytest.raises(ValueError):
        raise ValueError("error")

# 方法 2：匹配消息
def test_exception_message():
    with pytest.raises(ValueError, match="must be positive"):
        validate(-1)

# 方法 3：获取异常对象
def test_exception_info():
    with pytest.raises(ValueError) as exc_info:
        validate(-1)
    assert "positive" in str(exc_info.value)

# 方法 4：测试不抛出异常
def test_no_exception():
    try:
        validate(1)
    except ValueError:
        pytest.fail("不应该抛出异常")
```

---

## 9. conftest.py 的作用？

**答案**：

conftest.py 是 pytest 的特殊文件：

1. **共享 Fixture**：目录内所有测试可用
2. **钩子函数**：自定义 pytest 行为
3. **插件注册**：本地插件
4. **不需要导入**：自动加载

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def database():
    db = create_database()
    yield db
    db.close()

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 慢速测试")
```

**层级**：子目录的 conftest.py 可以覆盖父目录。

---

## 10. 如何并行运行测试？

**答案**：

使用 **pytest-xdist**：

```bash
# 安装
pip install pytest-xdist

# 自动检测 CPU 核心
pytest -n auto

# 指定进程数
pytest -n 4

# 按文件分发
pytest -n auto --dist=loadfile
```

**注意事项**：
1. 测试必须**独立**（无共享状态）
2. session 级 fixture 需小心
3. 数据库测试需要隔离
4. 输出可能交错

```python
# 确保测试独立
@pytest.fixture
def isolated_db():
    # 每个测试使用独立的数据库
    pass
```

