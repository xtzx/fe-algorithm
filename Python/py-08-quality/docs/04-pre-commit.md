# 04. Pre-commit - Git 钩子

## 本节目标

- 理解 pre-commit 的作用
- 配置常用钩子
- 与 CI 集成

---

## 什么是 Pre-commit

Pre-commit 是 Git 钩子管理框架：

```
类比 JavaScript:
pre-commit ≈ husky + lint-staged
```

**作用**：
- 提交前自动检查
- 保证代码质量
- 团队统一标准

---

## 安装和设置

```bash
# 安装
pip install pre-commit

# 初始化（创建 .pre-commit-config.yaml）
pre-commit sample-config > .pre-commit-config.yaml

# 安装 Git 钩子
pre-commit install

# 对所有文件运行
pre-commit run --all-files
```

---

## 配置文件

```yaml
# .pre-commit-config.yaml
repos:
  # Ruff - Linting 和格式化
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # 通用钩子
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## 常用钩子

### Ruff（推荐）

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.1.9
  hooks:
    - id: ruff
      args: [--fix]
    - id: ruff-format
```

### Black

```yaml
- repo: https://github.com/psf/black
  rev: 23.12.1
  hooks:
    - id: black
```

### Pyright

```yaml
- repo: https://github.com/RobertCraiworthy/pyright-python
  rev: v1.1.341
  hooks:
    - id: pyright
```

### MyPy

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.8.0
  hooks:
    - id: mypy
      additional_dependencies: [types-requests]
```

### 通用钩子

```yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.5.0
  hooks:
    # 文件检查
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-toml
    - id: check-json

    # 安全检查
    - id: check-added-large-files
    - id: check-merge-conflict
    - id: debug-statements

    # Python 特定
    - id: check-ast
    - id: check-docstring-first
```

---

## 与 husky + lint-staged 对比

| 功能 | pre-commit | husky + lint-staged |
|------|------------|---------------------|
| 配置文件 | .pre-commit-config.yaml | .husky/ + package.json |
| 钩子管理 | 内置 | 需要两个工具 |
| 工具隔离 | 自动管理 | 依赖 node_modules |
| 跨语言 | 原生支持 | 较复杂 |

### 配置对比

**pre-commit:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
```

**husky + lint-staged:**
```json
{
  "lint-staged": {
    "*.js": ["eslint --fix", "prettier --write"]
  }
}
```

---

## 高级配置

### 只检查暂存文件

默认行为：只检查 `git add` 的文件。

### 指定文件类型

```yaml
hooks:
  - id: mypy
    types: [python]
    files: ^src/
```

### 排除文件

```yaml
hooks:
  - id: ruff
    exclude: ^tests/fixtures/
```

### 额外依赖

```yaml
hooks:
  - id: mypy
    additional_dependencies:
      - types-requests
      - types-redis
```

### 传递参数

```yaml
hooks:
  - id: ruff
    args: [--fix, --select, "E,F,I"]
```

---

## 常用命令

```bash
# 安装钩子
pre-commit install

# 卸载钩子
pre-commit uninstall

# 手动运行所有钩子
pre-commit run --all-files

# 运行特定钩子
pre-commit run ruff --all-files

# 更新钩子版本
pre-commit autoupdate

# 跳过钩子提交（紧急情况）
git commit --no-verify
```

---

## CI 集成

### GitHub Actions

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: pre-commit/action@v3.0.0
```

### 自动更新

```yaml
# .github/workflows/pre-commit-autoupdate.yml
name: Pre-commit Autoupdate

on:
  schedule:
    - cron: '0 0 * * 0'  # 每周日

jobs:
  autoupdate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: |
          pip install pre-commit
          pre-commit autoupdate
      - uses: peter-evans/create-pull-request@v5
        with:
          title: Update pre-commit hooks
```

---

## 最佳实践

### 1. 最小化钩子

只包含必要的钩子，避免提交太慢。

### 2. 自动修复

使用 `--fix` 参数自动修复问题。

### 3. 定期更新

```bash
pre-commit autoupdate
```

### 4. CI 双重保障

本地 pre-commit + CI 检查。

---

## 本节要点

1. **pre-commit** 管理 Git 钩子
2. 等价于 **husky + lint-staged**
3. 使用 `pre-commit install` 安装
4. 配置在 `.pre-commit-config.yaml`
5. 支持自动修复和文件过滤
6. 与 CI 集成双重保障

