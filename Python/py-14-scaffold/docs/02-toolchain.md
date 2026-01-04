# 工具链配置

> 现代 Python 项目的工具链集成

## 1. 工具链概览

| 工具 | 用途 | 替代方案 |
|------|------|---------|
| **uv** | 包管理器 | pip, poetry, pdm |
| **ruff** | Lint + Format | flake8 + black + isort |
| **pyright** | 类型检查 | mypy |
| **pytest** | 测试 | unittest |
| **pre-commit** | Git Hooks | husky (JS) |

## 2. uv 工作流

### 安装 uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 常用命令

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# 安装依赖
uv pip install -r requirements.txt
uv pip install -e ".[dev]"

# 添加依赖
uv pip install requests

# 同步依赖
uv pip sync requirements.txt
```

### uv vs pip

```bash
# uv 比 pip 快 10-100x
time pip install numpy  # ~5s
time uv pip install numpy  # ~0.2s
```

## 3. Ruff 配置

### 基础配置

```toml
# pyproject.toml
[tool.ruff]
src = ["src"]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle
    "W",   # pycodestyle
    "F",   # Pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = ["E501"]
```

### 常用命令

```bash
# 检查代码
ruff check .

# 自动修复
ruff check --fix .

# 格式化代码
ruff format .

# 检查格式（不修改）
ruff format --check .
```

### Ruff vs 多工具

```bash
# 之前需要多个工具
flake8 .
black .
isort .

# 现在只需要 ruff
ruff check --fix .
ruff format .
```

## 4. Pyright 配置

### 基础配置

```toml
# pyproject.toml
[tool.pyright]
include = ["src"]
pythonVersion = "3.11"
typeCheckingMode = "basic"  # off, basic, standard, strict
```

### 类型检查模式

| 模式 | 严格程度 | 适用场景 |
|------|---------|---------|
| off | 关闭 | 不需要类型检查 |
| basic | 基础 | 新项目开始 |
| standard | 标准 | 大多数项目 |
| strict | 严格 | 类型完备的项目 |

### 常用命令

```bash
# 类型检查
pyright

# 指定目录
pyright src/

# 生成配置
pyright --createstub package_name
```

## 5. Pytest 配置

### 基础配置

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
]
```

### 常用命令

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_config.py

# 运行特定测试
pytest tests/test_config.py::test_settings

# 显示覆盖率
pytest --cov=src/scaffold --cov-report=term-missing

# 并行运行
pytest -n auto

# 运行标记的测试
pytest -m "not slow"
```

## 6. Pre-commit 配置

### 安装和设置

```bash
# 安装
pip install pre-commit

# 安装 hooks
pre-commit install

# 手动运行
pre-commit run --all-files

# 更新 hooks
pre-commit autoupdate
```

### 配置文件

```yaml
# .pre-commit-config.yaml
repos:
  # 通用检查
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml

  # Ruff
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

## 7. 脚本集合

### lint.sh

```bash
#!/bin/bash
set -e
echo "Running ruff check..."
ruff check .
echo "Running ruff format check..."
ruff format --check .
```

### format.sh

```bash
#!/bin/bash
set -e
echo "Running ruff format..."
ruff format .
echo "Running ruff check --fix..."
ruff check --fix .
```

### typecheck.sh

```bash
#!/bin/bash
set -e
echo "Running pyright..."
pyright
```

### test.sh

```bash
#!/bin/bash
set -e
echo "Running pytest..."
pytest tests/ -v --tb=short
```

### all.sh

```bash
#!/bin/bash
set -e
./scripts/lint.sh
./scripts/typecheck.sh
./scripts/test.sh
echo "All checks passed!"
```

## 8. CI/CD 集成

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Format
        run: ruff format --check .

      - name: Type check
        run: pyright

      - name: Test
        run: pytest --cov
```

## 9. VS Code 集成

### settings.json

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "python.analysis.typeCheckingMode": "basic"
}
```

### 推荐扩展

- Python (Microsoft)
- Ruff (Astral)
- Pylance (Microsoft)

