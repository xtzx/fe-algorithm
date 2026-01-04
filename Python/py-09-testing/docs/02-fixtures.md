# 02. Fixture 机制

## 本节目标

- 理解 fixture 的作用和生命周期
- 掌握内置 fixture
- 使用 conftest.py 共享 fixture

---

## 什么是 Fixture

Fixture 是测试的准备和清理机制：

```
类比 JavaScript:
fixture ≈ beforeEach + afterEach + 依赖注入
```

---

## 基本 Fixture

```python
import pytest

@pytest.fixture
def sample_data():
    """返回测试数据"""
    return {"name": "Alice", "age": 30}

def test_name(sample_data):
    assert sample_data["name"] == "Alice"

def test_age(sample_data):
    assert sample_data["age"] == 30
```

---

## Fixture 带清理

```python
import pytest

@pytest.fixture
def database_connection():
    """创建数据库连接，测试后关闭"""
    # Setup
    conn = create_connection()

    yield conn  # 提供给测试使用

    # Teardown
    conn.close()

def test_query(database_connection):
    result = database_connection.execute("SELECT 1")
    assert result == 1
```

---

## Fixture Scope

```python
@pytest.fixture(scope="function")  # 默认：每个测试函数
def func_fixture():
    return create_resource()

@pytest.fixture(scope="class")  # 每个测试类
def class_fixture():
    return create_resource()

@pytest.fixture(scope="module")  # 每个测试模块
def module_fixture():
    return create_resource()

@pytest.fixture(scope="session")  # 整个测试会话
def session_fixture():
    return create_resource()
```

### Scope 对比

| Scope | 创建次数 | 适用场景 |
|-------|----------|----------|
| function | 每个测试 | 需要隔离的数据 |
| class | 每个类 | 类内共享 |
| module | 每个文件 | 文件内共享 |
| session | 整个会话 | 昂贵资源（数据库连接） |

---

## conftest.py

`conftest.py` 用于共享 fixture：

```
tests/
├── conftest.py           # 所有测试共享
├── unit/
│   ├── conftest.py       # unit 目录共享
│   └── test_math.py
└── integration/
    ├── conftest.py       # integration 目录共享
    └── test_api.py
```

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def app_config():
    return {"debug": True, "db_url": "sqlite:///:memory:"}

@pytest.fixture
def client(app_config):
    return create_client(app_config)
```

---

## 内置 Fixture

### tmp_path - 临时目录

```python
def test_create_file(tmp_path):
    """tmp_path 提供临时目录"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    assert file_path.read_text() == "hello"
    # 测试后自动清理
```

### tmp_path_factory - 会话级临时目录

```python
@pytest.fixture(scope="session")
def shared_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")
```

### monkeypatch - 修改环境

```python
def test_env_variable(monkeypatch):
    """修改环境变量"""
    monkeypatch.setenv("API_KEY", "test-key")
    assert os.environ["API_KEY"] == "test-key"

def test_modify_attribute(monkeypatch):
    """修改对象属性"""
    monkeypatch.setattr("module.CONFIG", {"debug": True})

def test_delete_attribute(monkeypatch):
    """删除属性"""
    monkeypatch.delattr("module.OPTIONAL_FEATURE")

def test_change_dict(monkeypatch):
    """修改字典"""
    monkeypatch.setitem(my_dict, "key", "new_value")
```

### capsys - 捕获输出

```python
def test_print(capsys):
    """捕获 stdout/stderr"""
    print("hello")
    print("error", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert captured.err == "error\n"
```

### caplog - 捕获日志

```python
import logging

def test_logging(caplog):
    """捕获日志"""
    with caplog.at_level(logging.INFO):
        logging.info("test message")

    assert "test message" in caplog.text
    assert len(caplog.records) == 1
```

---

## Fixture 依赖

```python
@pytest.fixture
def user():
    return User(name="Alice")

@pytest.fixture
def authenticated_user(user):
    """依赖 user fixture"""
    user.authenticate()
    return user

def test_user_action(authenticated_user):
    authenticated_user.do_something()
```

---

## Fixture 参数化

```python
@pytest.fixture(params=["mysql", "postgresql", "sqlite"])
def database(request):
    """参数化 fixture"""
    db_type = request.param
    db = create_database(db_type)
    yield db
    db.close()

def test_insert(database):
    # 会运行 3 次，每种数据库类型一次
    database.insert({"key": "value"})
```

---

## autouse - 自动使用

```python
@pytest.fixture(autouse=True)
def reset_state():
    """每个测试前自动执行"""
    global_state.reset()
    yield
    global_state.cleanup()

def test_something():
    # reset_state 自动在测试前后执行
    pass
```

---

## Fixture 工厂

```python
@pytest.fixture
def make_user():
    """工厂 fixture"""
    created_users = []

    def _make_user(name, age=25):
        user = User(name=name, age=age)
        created_users.append(user)
        return user

    yield _make_user

    # 清理所有创建的用户
    for user in created_users:
        user.delete()

def test_multiple_users(make_user):
    alice = make_user("Alice", 30)
    bob = make_user("Bob", 25)
    assert alice.name != bob.name
```

---

## 对比 Jest

**Jest:**
```javascript
let user;

beforeEach(() => {
    user = createUser();
});

afterEach(() => {
    user.cleanup();
});

test('user name', () => {
    expect(user.name).toBe('Alice');
});
```

**pytest:**
```python
@pytest.fixture
def user():
    user = create_user()
    yield user
    user.cleanup()

def test_user_name(user):
    assert user.name == "Alice"
```

---

## 本节要点

1. **fixture** 是测试的设置/清理机制
2. **yield** 分隔设置和清理
3. **scope** 控制生命周期
4. **conftest.py** 共享 fixture
5. **内置 fixture**: tmp_path, monkeypatch, capsys
6. **参数化 fixture**: `params` 参数

