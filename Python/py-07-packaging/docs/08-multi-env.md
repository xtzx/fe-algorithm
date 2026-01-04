# 08. 多环境管理

## 本节目标

- 管理开发、测试、生产依赖
- 使用 optional dependencies
- CI/CD 最佳实践

---

## 依赖分组

### pyproject.toml 方式

```toml
[project]
name = "myproject"
version = "0.1.0"
dependencies = [
    # 生产依赖
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    # 开发依赖
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
test = [
    # 测试依赖
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "httpx>=0.24",  # 测试 FastAPI
]
docs = [
    # 文档依赖
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
]
```

### 安装方式

```bash
# 只安装生产依赖
pip install .

# 安装开发依赖
pip install ".[dev]"

# 安装多个组
pip install ".[dev,test,docs]"

# 可编辑安装 + 开发依赖
pip install -e ".[dev]"
```

---

## poetry 依赖组

```toml
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.100"
uvicorn = "^0.23"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
ruff = "^0.1"

[tool.poetry.group.test.dependencies]
pytest-asyncio = "^0.21"
httpx = "^0.24"

[tool.poetry.group.docs]
optional = true  # 可选组

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.5"
```

### 安装

```bash
# 所有非可选组
poetry install

# 包含可选组
poetry install --with docs

# 只安装特定组
poetry install --only main

# 排除组
poetry install --without dev
```

---

## 环境变量区分

### .env 文件

```bash
# .env.development
DEBUG=true
DATABASE_URL=sqlite:///dev.db

# .env.production
DEBUG=false
DATABASE_URL=postgresql://...

# .env.test
DEBUG=true
DATABASE_URL=sqlite:///test.db
```

### 代码中使用

```python
import os
from dotenv import load_dotenv

# 根据环境加载
env = os.getenv("ENV", "development")
load_dotenv(f".env.{env}")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")
```

---

## 多 requirements 文件

### 传统方式

```
requirements/
├── base.txt        # 基础依赖
├── dev.txt         # 开发依赖
├── test.txt        # 测试依赖
└── prod.txt        # 生产依赖
```

```txt
# requirements/base.txt
fastapi>=0.100
uvicorn>=0.23

# requirements/dev.txt
-r base.txt
pytest>=7.0
ruff>=0.1

# requirements/test.txt
-r base.txt
pytest>=7.0
httpx>=0.24

# requirements/prod.txt
-r base.txt
gunicorn>=21.0
```

### 使用

```bash
pip install -r requirements/dev.txt
```

---

## 锁定多环境

### uv 方式

```bash
# 生产环境
uv pip compile pyproject.toml -o requirements.lock

# 开发环境
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock

# 测试环境
uv pip compile pyproject.toml --extra test -o requirements-test.lock
```

### poetry 方式

```bash
# 导出不同环境
poetry export -f requirements.txt -o requirements.txt --without dev

poetry export -f requirements.txt -o requirements-dev.txt --with dev
```

---

## CI/CD 配置

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-${{ hashFiles('requirements-test.lock') }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip sync requirements-test.lock

      - name: Run tests
        run: pytest

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: |
          pip install build
          python -m build

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/
```

### Docker 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.12-slim as builder

WORKDIR /app
COPY pyproject.toml requirements.lock ./
RUN pip install uv && uv pip install --system -r requirements.lock

# 运行阶段
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src/ ./src/

CMD ["python", "-m", "myapp"]
```

---

## 缓存策略

### 基于 lockfile 哈希

```yaml
- name: Cache
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/uv
    key: ${{ runner.os }}-python-${{ hashFiles('requirements.lock') }}
    restore-keys: |
      ${{ runner.os }}-python-
```

### 分层缓存

```yaml
- name: Cache base dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-base-${{ hashFiles('requirements.lock') }}

- name: Cache dev dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-dev-${{ hashFiles('requirements-dev.lock') }}
```

---

## 环境隔离检查

### 验证安装

```bash
# 检查依赖冲突
pip check

# 查看已安装包
pip list

# 查看依赖树
pip install pipdeptree
pipdeptree
```

### 清洁安装测试

```bash
# 创建全新环境
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# 从锁定文件安装
uv pip sync requirements.lock

# 运行测试
pytest
```

---

## 本节要点

1. **optional-dependencies** 分组依赖
2. **pip install ".[dev]"** 安装特定组
3. **多 lockfile** 锁定不同环境
4. **CI 缓存** 基于 lockfile 哈希
5. **Docker** 多阶段构建优化

