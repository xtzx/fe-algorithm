# 01. pyproject.toml 详解

## 本节目标

- 理解 pyproject.toml 的结构和作用
- 掌握项目元数据和依赖声明
- 对比 package.json

---

## 什么是 pyproject.toml

`pyproject.toml` 是 Python 项目的统一配置文件，类似于 Node.js 的 `package.json`。

**PEP 标准**：
- **PEP 517**: 构建系统接口
- **PEP 518**: 构建依赖声明
- **PEP 621**: 项目元数据标准

---

## 基本结构

```toml
# 项目元数据
[project]
name = "my-awesome-project"
version = "0.1.0"
description = "A short description"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["python", "example"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.12",
]

# 核心依赖
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0,<3.0",
]

# 可选依赖
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
]

# CLI 入口
[project.scripts]
mycli = "my_project.cli:main"

# 项目 URL
[project.urls]
Homepage = "https://github.com/user/project"
Documentation = "https://project.readthedocs.io"

# 构建系统
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 与 package.json 对比

| 字段 | pyproject.toml | package.json |
|------|----------------|--------------|
| 名称 | `name` | `name` |
| 版本 | `version` | `version` |
| 描述 | `description` | `description` |
| 入口 | `project.scripts` | `bin` |
| 依赖 | `dependencies` | `dependencies` |
| 开发依赖 | `optional-dependencies.dev` | `devDependencies` |
| Python/Node 版本 | `requires-python` | `engines.node` |
| 脚本 | 无（用 Makefile） | `scripts` |

### 示例对照

**pyproject.toml**:
```toml
[project]
name = "myapp"
version = "1.0.0"
dependencies = ["requests>=2.28"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
myapp = "myapp.cli:main"
```

**package.json**:
```json
{
  "name": "myapp",
  "version": "1.0.0",
  "dependencies": {
    "axios": "^1.0.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  },
  "bin": {
    "myapp": "./bin/cli.js"
  }
}
```

---

## 依赖版本语法

```toml
dependencies = [
    # 精确版本
    "package==1.0.0",

    # 最小版本
    "package>=1.0.0",

    # 兼容版本（~= 等于 >=1.0.0,<2.0.0）
    "package~=1.0",

    # 范围
    "package>=1.0.0,<2.0.0",

    # 排除版本
    "package>=1.0.0,!=1.5.0",

    # 预发布版本
    "package>=1.0.0a1",

    # 带 extras
    "package[extra1,extra2]>=1.0",

    # Git 依赖
    "package @ git+https://github.com/user/repo.git@main",

    # 本地路径
    "package @ file:///path/to/package",
]
```

---

## 构建系统

### 常见 build backend

```toml
# Hatchling（推荐，现代）
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Setuptools（传统，广泛支持）
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

# Poetry
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Flit（简单项目）
[build-system]
requires = ["flit_core>=3.4"]
build-backend = "flit_core.buildapi"
```

### 构建系统对比

| Backend | 特点 | 适用场景 |
|---------|------|----------|
| hatchling | 现代、快速 | 新项目 |
| setuptools | 广泛兼容 | 复杂构建 |
| poetry-core | 集成 poetry | poetry 用户 |
| flit | 极简 | 纯 Python 包 |

---

## 工具配置

pyproject.toml 也是各种工具的配置中心：

```toml
# pytest 配置
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

# ruff 配置（linter）
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I"]

# mypy 配置（类型检查）
[tool.mypy]
python_version = "3.12"
strict = true

# black 配置（格式化）
[tool.black]
line-length = 88
target-version = ["py312"]

# coverage 配置
[tool.coverage.run]
source = ["src"]
branch = true

# hatch 配置
[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

---

## src 布局 vs flat 布局

### src 布局（推荐）

```
project/
├── pyproject.toml
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── main.py
└── tests/
    └── test_main.py
```

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]

# 或 setuptools
[tool.setuptools.packages.find]
where = ["src"]
```

### flat 布局

```
project/
├── pyproject.toml
├── mypackage/
│   ├── __init__.py
│   └── main.py
└── tests/
    └── test_main.py
```

**src 布局优势**：
- 避免导入冲突（不会意外导入源码目录）
- 强制使用 editable install
- 更清晰的结构

---

## 动态字段

某些字段可以动态生成：

```toml
[project]
name = "mypackage"
dynamic = ["version"]

[tool.hatch.version]
path = "src/mypackage/__init__.py"
```

```python
# src/mypackage/__init__.py
__version__ = "1.0.0"
```

---

## 完整示例

```toml
[project]
name = "my-awesome-app"
version = "0.1.0"
description = "An awesome application"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [
    { name = "Developer", email = "dev@example.com" }
]

dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0",
    "sqlalchemy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[project.scripts]
myapp = "my_awesome_app.cli:main"

[project.urls]
Homepage = "https://github.com/user/my-awesome-app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_awesome_app"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
```

---

## 本节要点

1. **pyproject.toml** 是 Python 项目的统一配置文件
2. **[project]** 声明元数据和依赖
3. **[build-system]** 指定构建后端
4. **[tool.*]** 配置各种开发工具
5. **src 布局** 是推荐的项目结构
6. 类似 package.json，但更标准化

