# P09: 测试体系

> 建立 pytest 思维、mock 技巧、覆盖率意识

## 学完后能做

- 编写 pytest 测试
- 合理使用 mock
- 理解测试金字塔

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 带覆盖率
pytest --cov=src --cov-report=term-missing

# 运行特定测试
pytest tests/test_math_utils.py -v
```

## Python vs JavaScript 测试对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| 测试框架 | pytest | Jest / Vitest |
| 断言 | assert | expect() |
| Mock | unittest.mock | jest.mock() |
| Fixture | @pytest.fixture | beforeEach |
| 覆盖率 | coverage.py | istanbul |
| 配置文件 | pyproject.toml | jest.config.js |

## 目录结构

```
py-09-testing/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-pytest-basics.md      # pytest 基础
│   ├── 02-fixtures.md           # fixture 机制
│   ├── 03-mock.md               # mock 策略
│   ├── 04-coverage.md           # 覆盖率
│   ├── 05-integration.md        # 集成测试
│   ├── 06-organization.md       # 测试组织
│   ├── 07-exercises.md          # 练习题
│   └── 08-interview-questions.md # 面试题
├── src/testing_lab/
│   ├── math_utils.py            # 数学工具
│   ├── file_utils.py            # 文件工具
│   └── http_utils.py            # HTTP 工具
├── tests/
│   ├── conftest.py              # 共享 fixture
│   ├── test_math_utils.py
│   ├── test_file_utils.py
│   └── test_http_utils.py
└── scripts/
```

## 测试金字塔

```
         ╱╲
        ╱  ╲
       ╱ E2E╲        少量端到端测试
      ╱──────╲
     ╱ 集成   ╲      适量集成测试
    ╱──────────╲
   ╱   单元     ╲    大量单元测试
  ╱──────────────╲
```

## 核心概念速查

### pytest 命令

```bash
pytest                          # 运行所有测试
pytest tests/test_file.py       # 运行特定文件
pytest -k "test_add"            # 按名称匹配
pytest -m slow                  # 按标记筛选
pytest -v                       # 详细输出
pytest -x                       # 失败即停
pytest --pdb                    # 失败时调试
pytest -n auto                  # 并行运行
```

### fixture scope

```python
@pytest.fixture(scope="function")  # 每个测试函数（默认）
@pytest.fixture(scope="class")     # 每个测试类
@pytest.fixture(scope="module")    # 每个模块
@pytest.fixture(scope="session")   # 整个测试会话
```

### mock 模式

```python
from unittest.mock import patch, MagicMock

# 装饰器
@patch("module.function")
def test_something(mock_func):
    mock_func.return_value = 42

# 上下文管理器
with patch("module.function") as mock_func:
    mock_func.return_value = 42
```

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| mock 路径错误 | patch 的是导入路径 | patch 使用处，非定义处 |
| fixture 不共享 | 默认 function scope | 根据需要调整 scope |
| 测试顺序依赖 | 测试应该独立 | 每个测试独立设置/清理 |
| 过度 mock | 测试没意义 | 只 mock 外部依赖 |

## 学习路径

1. [pytest 基础](docs/01-pytest-basics.md)
2. [fixture 机制](docs/02-fixtures.md)
3. [mock 策略](docs/03-mock.md)
4. [覆盖率](docs/04-coverage.md)
5. [集成测试](docs/05-integration.md)
6. [测试组织](docs/06-organization.md)

