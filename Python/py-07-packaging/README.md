# P07: 包与环境管理

> 像管理 Node 依赖一样管理 Python：可重复、可锁定、可迁移

## 学完后能做

- 正确配置 pyproject.toml
- 使用 uv/poetry 管理依赖
- 搭建可重复的开发环境

## 快速开始

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 或使用 pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 目录结构

```
py-07-packaging/
├── README.md
├── pyproject.toml           # 项目配置（类似 package.json）
├── docs/
│   ├── 01-pyproject-toml.md     # pyproject.toml 详解
│   ├── 02-venv.md               # 虚拟环境
│   ├── 03-package-formats.md    # wheel vs sdist
│   ├── 04-dependency-tools.md   # uv/poetry/pdm 对比
│   ├── 05-lockfile.md           # lockfile 最佳实践
│   ├── 06-entry-points.md       # CLI 入口
│   ├── 07-private-index.md      # 私有源配置
│   ├── 08-multi-env.md          # 多环境管理
│   ├── 09-exercises.md          # 练习题
│   └── 10-interview-questions.md # 面试题
├── src/
│   └── packaging_lab/       # 示例包
├── tests/
└── scripts/
```

## Python vs Node.js 包管理对比

| 概念 | Python | Node.js |
|------|--------|---------|
| 项目配置 | `pyproject.toml` | `package.json` |
| 依赖锁定 | `uv.lock` / `poetry.lock` | `package-lock.json` |
| 包管理器 | pip / uv / poetry | npm / yarn / pnpm |
| 虚拟环境 | venv | node_modules |
| 包仓库 | PyPI | npm registry |
| 包格式 | wheel / sdist | tarball |
| CLI 入口 | entry_points | bin |

## 核心概念速查

### pyproject.toml 基本结构

```toml
[project]
name = "myproject"
version = "0.1.0"
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[project.scripts]
mycli = "myproject.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 工具选择指南

| 场景 | 推荐工具 |
|------|----------|
| 新项目 | uv |
| 需要 workspace | poetry |
| 简单脚本 | pip + venv |
| CI/CD | uv（快速） |

### 常用命令

```bash
# uv
uv venv                    # 创建虚拟环境
uv pip install package     # 安装包
uv pip compile pyproject.toml -o requirements.lock  # 锁定
uv pip sync requirements.lock  # 同步

# poetry
poetry init               # 初始化项目
poetry add package        # 添加依赖
poetry lock               # 锁定依赖
poetry install            # 安装依赖

# pip
pip install -e ".[dev]"   # 可编辑安装
pip freeze > requirements.txt  # 导出依赖
```

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 依赖漂移 | 没有 lockfile | 使用 uv.lock 或 poetry.lock |
| 环境污染 | 全局安装 | 始终使用虚拟环境 |
| 版本冲突 | 依赖树冲突 | 使用 `pip check` 检查 |
| editable 问题 | 修改不生效 | 确保 `pip install -e .` |

## 学习路径

1. [pyproject.toml 详解](docs/01-pyproject-toml.md)
2. [虚拟环境](docs/02-venv.md)
3. [包格式](docs/03-package-formats.md)
4. [依赖管理工具](docs/04-dependency-tools.md)
5. [lockfile 最佳实践](docs/05-lockfile.md)
6. [CLI 入口](docs/06-entry-points.md)
7. [私有源配置](docs/07-private-index.md)
8. [多环境管理](docs/08-multi-env.md)

