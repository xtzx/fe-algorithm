# 06. 测试组织

## 本节目标

- 组织测试目录结构
- 使用标记分类测试
- 并行运行测试

---

## 目录结构

### 推荐结构

```
project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── models.py
│       ├── services.py
│       └── utils.py
├── tests/
│   ├── conftest.py          # 共享 fixture
│   ├── unit/                 # 单元测试
│   │   ├── conftest.py
│   │   ├── test_models.py
│   │   ├── test_services.py
│   │   └── test_utils.py
│   ├── integration/          # 集成测试
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── e2e/                  # 端到端测试
│       └── test_workflows.py
└── pyproject.toml
```

### 简单结构

```
project/
├── src/
│   └── mypackage/
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
└── pyproject.toml
```

---

## 命名约定

| 元素 | 约定 | 示例 |
|------|------|------|
| 测试文件 | test_*.py | test_user.py |
| 测试函数 | test_* | test_create_user |
| 测试类 | Test* | TestUserService |
| Fixture | 描述性名称 | db_session, client |

```python
# test_user.py

class TestUserModel:
    """用户模型测试"""

    def test_create_user(self):
        """测试创建用户"""
        pass

    def test_user_validation(self):
        """测试用户验证"""
        pass

class TestUserService:
    """用户服务测试"""

    def test_get_user_by_id(self):
        pass

    def test_list_users(self):
        pass
```

---

## 标记（Markers）

### 定义标记

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: 标记为慢速测试",
    "integration: 集成测试",
    "unit: 单元测试",
    "smoke: 冒烟测试",
]
```

### 使用标记

```python
import pytest

@pytest.mark.unit
def test_fast():
    assert 1 + 1 == 2

@pytest.mark.slow
def test_slow():
    import time
    time.sleep(5)
    assert True

@pytest.mark.integration
def test_database():
    # 需要数据库
    pass

# 多个标记
@pytest.mark.integration
@pytest.mark.slow
def test_external_api():
    pass
```

### 按标记运行

```bash
# 运行所有单元测试
pytest -m unit

# 运行非慢速测试
pytest -m "not slow"

# 组合条件
pytest -m "unit and not slow"
pytest -m "unit or integration"
```

---

## conftest.py

### 作用

1. **共享 fixture**：目录内所有测试可用
2. **钩子函数**：自定义 pytest 行为
3. **插件**：注册本地插件

### 层级

```
tests/
├── conftest.py           # 所有测试共享
├── unit/
│   ├── conftest.py       # 只在 unit/ 内共享
│   └── test_*.py
└── integration/
    ├── conftest.py       # 只在 integration/ 内共享
    └── test_*.py
```

### 示例

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def app_config():
    """应用配置（会话级）"""
    return {"debug": True}

@pytest.fixture
def client(app_config):
    """测试客户端"""
    return create_client(app_config)

# tests/integration/conftest.py
import pytest

@pytest.fixture(scope="module")
def db_connection():
    """数据库连接（模块级）"""
    conn = create_connection()
    yield conn
    conn.close()
```

---

## 并行运行

### 使用 pytest-xdist

```bash
pip install pytest-xdist
```

```bash
# 自动检测 CPU 核心数
pytest -n auto

# 指定进程数
pytest -n 4

# 按文件分发
pytest -n auto --dist=loadfile
```

### 分发策略

| 策略 | 说明 |
|------|------|
| load | 按负载分发（默认） |
| loadscope | 按 scope 分组 |
| loadfile | 按文件分组 |
| no | 不分发（调试用） |

### 注意事项

```python
# 并行测试需要注意：
# 1. 避免共享状态
# 2. 使用 scope="session" 的 fixture 小心
# 3. 数据库测试需要隔离
```

---

## 测试选择

```bash
# 按文件
pytest tests/test_user.py

# 按类
pytest tests/test_user.py::TestUserModel

# 按函数
pytest tests/test_user.py::TestUserModel::test_create

# 按名称模式
pytest -k "user"
pytest -k "user and not delete"
pytest -k "test_create or test_update"

# 按标记
pytest -m integration
pytest -m "not slow"
```

---

## 测试输出

```bash
# 详细输出
pytest -v

# 更详细
pytest -vv

# 显示 print 输出
pytest -s

# 简短 traceback
pytest --tb=short

# 只显示失败
pytest --tb=no

# 进度条
pytest --progress
```

---

## 失败处理

```bash
# 失败即停
pytest -x

# N 个失败后停止
pytest --maxfail=3

# 只运行上次失败的
pytest --lf

# 先运行上次失败的
pytest --ff

# 失败时进入调试器
pytest --pdb
```

---

## CI 配置

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests
        run: pytest -m unit --cov=src

      - name: Run integration tests
        run: pytest -m integration
```

---

## 本节要点

1. **目录结构**：unit/integration/e2e 分开
2. **命名约定**：test_*.py，test_* 函数
3. **标记**：分类测试，按需运行
4. **conftest.py**：共享 fixture
5. **并行**：pytest-xdist 加速
6. **选择**：-k 按名称，-m 按标记

