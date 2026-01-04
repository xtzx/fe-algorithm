# P14: 工程化脚手架

> 可复用的 Python 项目模板，快速初始化规范项目

## 🎯 学习目标

完成本模块后，你将能够：

- 快速初始化规范的 Python 项目
- 统一团队工程规范
- 掌握现代 Python 项目的最佳实践

## 📋 前置要求

- 完成 P13（文件自动化）
- 了解 `pyproject.toml` 配置
- 熟悉 `ruff`、`pytest`、`pre-commit`

## 🗺️ 知识图谱

```
工程化脚手架
├── 项目结构
│   ├── src 布局
│   ├── 配置管理
│   ├── 日志配置
│   └── CLI 入口
├── 工具链集成
│   ├── pyproject.toml 完整配置
│   ├── uv 工作流
│   ├── ruff + pyright
│   ├── pytest
│   └── pre-commit
├── 常用模式
│   ├── pydantic-settings + .env
│   ├── 日志初始化
│   └── CLI 框架
└── 脚本集合
    ├── lint
    ├── format
    ├── typecheck
    ├── test
    └── run
```

## 🚀 快速开始

### 使用模板创建新项目

```bash
# 方法 1: 直接复制模板
cp -r py-14-scaffold my-new-project
cd my-new-project

# 方法 2: 使用脚本生成
./scripts/create-project.sh my-new-project

# 设置环境
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 复制环境配置
cp .env.example .env

# 安装 pre-commit hooks
pre-commit install

# 运行测试
pytest

# 运行应用
python -m scaffold --help
```

## 📁 项目结构

```
py-14-scaffold/
├── README.md                 # 项目说明
├── pyproject.toml           # 项目配置（依赖、工具链）
├── .python-version          # Python 版本锁定
├── .env.example             # 环境变量示例
├── .pre-commit-config.yaml  # Pre-commit 配置
├── docs/                    # 文档
│   ├── 01-project-structure.md   # 项目结构说明
│   ├── 02-toolchain.md           # 工具链配置
│   └── 03-patterns.md            # 常用模式
├── src/                     # 源码目录
│   └── scaffold/            # 主包
│       ├── __init__.py          # 包初始化
│       ├── __main__.py          # python -m 入口
│       ├── cli.py               # CLI 命令
│       ├── config.py            # 配置管理
│       ├── log.py               # 日志配置
│       └── utils.py             # 工具函数
├── tests/                   # 测试
│   ├── __init__.py
│   ├── conftest.py              # 共享 fixtures
│   └── test_config.py
├── examples/                # 示例
│   └── sample_usage.py
└── scripts/                 # 脚本
    ├── lint.sh                  # 代码检查
    ├── format.sh                # 代码格式化
    ├── typecheck.sh             # 类型检查
    ├── test.sh                  # 运行测试
    ├── run.sh                   # 运行应用
    └── create-project.sh        # 创建新项目
```

## 🔧 工具链概览

| 工具 | 用途 | 配置位置 |
|------|------|---------|
| **uv** | 包管理器（替代 pip） | - |
| **ruff** | Linting + Formatting | `pyproject.toml` |
| **pyright** | 类型检查 | `pyproject.toml` |
| **pytest** | 测试框架 | `pyproject.toml` |
| **pre-commit** | Git hooks | `.pre-commit-config.yaml` |

## 📝 配置管理

使用 `pydantic-settings` + `.env` 文件：

```python
from scaffold.config import get_settings

settings = get_settings()
print(settings.app_name)
print(settings.debug)
print(settings.database_url)
```

`.env` 文件示例：

```env
APP_NAME=my-app
DEBUG=true
DATABASE_URL=postgresql://localhost/mydb
```

## 📊 日志配置

统一的日志初始化：

```python
from scaffold.log import setup_logging, get_logger

# 初始化（通常在入口处调用一次）
setup_logging(level="INFO", json_format=False)

# 获取 logger
logger = get_logger(__name__)
logger.info("Application started")
```

## 🖥️ CLI 框架

基于 `argparse` 的 CLI 模板：

```bash
# 运行帮助
python -m scaffold --help

# 运行命令
python -m scaffold run --config config.toml
python -m scaffold version
```

## 🔑 核心特性

### 1. src 布局

```
project/
├── src/
│   └── my_package/
│       └── ...
└── tests/
```

优势：
- 明确区分源码和测试
- 避免导入混乱
- 更好的打包体验

### 2. 统一工具配置

所有工具配置集中在 `pyproject.toml`：

```toml
[tool.ruff]
line-length = 88

[tool.pyright]
typeCheckingMode = "basic"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 3. 开发工作流

```bash
# 格式化代码
./scripts/format.sh

# 代码检查
./scripts/lint.sh

# 类型检查
./scripts/typecheck.sh

# 运行测试
./scripts/test.sh

# 全部检查
./scripts/lint.sh && ./scripts/typecheck.sh && ./scripts/test.sh
```

## ⚡ JS/TS 工程师对照

| Python | JS/TS 类比 |
|--------|-----------|
| `pyproject.toml` | `package.json` |
| `uv` | `pnpm` / `bun` |
| `ruff` | `eslint` + `prettier` |
| `pyright` | `tsc --noEmit` |
| `pytest` | `jest` / `vitest` |
| `pre-commit` | `husky` + `lint-staged` |
| `.env` + `pydantic-settings` | `dotenv` + `zod` |

## ✅ 模板特性检查清单

- [x] src 布局项目结构
- [x] `pyproject.toml` 完整配置
- [x] `.python-version` 版本锁定
- [x] `.env.example` 环境变量模板
- [x] `pydantic-settings` 配置管理
- [x] 统一日志配置
- [x] CLI 入口模板
- [x] `ruff` lint + format
- [x] `pyright` 类型检查
- [x] `pytest` 测试配置
- [x] `pre-commit` hooks
- [x] 开发脚本集合

## 🔗 延伸阅读

- [Python Packaging User Guide](https://packaging.python.org/)
- [uv 文档](https://github.com/astral-sh/uv)
- [ruff 文档](https://docs.astral.sh/ruff/)
- [pydantic-settings 文档](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

