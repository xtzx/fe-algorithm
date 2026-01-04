# 项目结构说明

> src 布局的现代 Python 项目结构

## 1. 为什么使用 src 布局？

### 传统布局

```
my_project/
├── my_package/
│   ├── __init__.py
│   └── module.py
├── tests/
│   └── test_module.py
└── setup.py
```

问题：
- 运行测试时可能意外导入本地包而非安装的包
- 开发和安装环境行为不一致

### src 布局

```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
├── tests/
│   └── test_module.py
└── pyproject.toml
```

优势：
- ✅ 明确区分源码和测试
- ✅ 强制安装后才能导入
- ✅ 开发和生产环境一致
- ✅ 现代工具的推荐方式

## 2. 标准目录结构

```
project-name/
├── .github/                 # GitHub Actions (可选)
│   └── workflows/
│       └── ci.yml
├── docs/                    # 文档
│   ├── api/
│   └── guides/
├── examples/                # 示例代码
├── scripts/                 # 开发脚本
│   ├── lint.sh
│   ├── format.sh
│   ├── test.sh
│   └── run.sh
├── src/                     # 源码
│   └── package_name/
│       ├── __init__.py      # 包初始化，导出公共 API
│       ├── __main__.py      # python -m 入口
│       ├── cli.py           # CLI 命令
│       ├── config.py        # 配置管理
│       ├── log.py           # 日志配置
│       ├── core/            # 核心功能
│       │   ├── __init__.py
│       │   └── ...
│       └── utils/           # 工具函数
│           ├── __init__.py
│           └── ...
├── tests/                   # 测试
│   ├── __init__.py
│   ├── conftest.py          # 共享 fixtures
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   └── e2e/                 # 端到端测试
├── .env.example             # 环境变量示例
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version          # Python 版本
├── pyproject.toml           # 项目配置
├── README.md
└── LICENSE
```

## 3. 关键文件说明

### `__init__.py`

定义包的公共 API：

```python
"""
Package description
"""

from package_name.module import func1, func2
from package_name.config import Settings

__version__ = "0.1.0"

__all__ = [
    "func1",
    "func2",
    "Settings",
    "__version__",
]
```

### `__main__.py`

支持 `python -m package_name`：

```python
"""Module entry point"""

from package_name.cli import main

if __name__ == "__main__":
    main()
```

### `conftest.py`

共享 pytest fixtures：

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_dir(tmp_path):
    """临时目录 fixture"""
    return tmp_path

@pytest.fixture
def sample_data():
    """示例数据 fixture"""
    return {"key": "value"}
```

## 4. 命名规范

### 包和模块

```python
# ✅ 好的命名
my_package/
    data_processing.py
    file_utils.py
    http_client.py

# ❌ 避免
MyPackage/           # 使用下划线，不用驼峰
data-processing.py   # 使用下划线，不用连字符
```

### 类和函数

```python
# 类: PascalCase
class DataProcessor:
    pass

# 函数/方法: snake_case
def process_data():
    pass

# 常量: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3

# 私有: 前缀下划线
def _internal_helper():
    pass
```

## 5. 可编辑安装

开发时使用可编辑安装：

```bash
# 使用 pip
pip install -e ".[dev]"

# 使用 uv (推荐)
uv pip install -e ".[dev]"
```

这样：
- 代码修改立即生效
- 可以正常导入包
- 测试使用安装的包

## 6. 与 JS/TS 项目对比

| Python | JS/TS |
|--------|-------|
| `src/package_name/` | `src/` |
| `tests/` | `tests/` 或 `__tests__/` |
| `pyproject.toml` | `package.json` |
| `__init__.py` | `index.ts` |
| `__main__.py` | `bin` 字段 |
| `.env.example` | `.env.example` |

## 7. 最佳实践

### 分层结构

```
src/my_app/
├── api/          # API 层（路由、控制器）
├── core/         # 核心业务逻辑
├── models/       # 数据模型
├── services/     # 服务层
├── repositories/ # 数据访问层
└── utils/        # 工具函数
```

### 避免循环导入

```python
# ❌ 循环导入
# a.py
from b import func_b

# b.py
from a import func_a  # 循环！

# ✅ 解决方案
# 1. 重构设计，减少依赖
# 2. 延迟导入（在函数内导入）
# 3. 使用 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a import SomeType
```

### 相对导入 vs 绝对导入

```python
# 推荐：绝对导入
from my_package.utils import helper

# 谨慎使用：相对导入
from .utils import helper
from ..core import engine
```

