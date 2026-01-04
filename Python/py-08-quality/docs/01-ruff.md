# 01. Ruff - 现代 Python Linter

## 本节目标

- 理解 ruff 的功能和优势
- 掌握常用规则配置
- 替代 flake8 + isort + 部分 pylint

---

## 什么是 Ruff

Ruff 是用 Rust 编写的超快 Python linter，可替代多个工具：

| 替代工具 | 功能 |
|----------|------|
| Flake8 | 代码检查 |
| isort | import 排序 |
| pydocstyle | 文档检查 |
| pyupgrade | 代码现代化 |
| autoflake | 移除无用代码 |

**速度**: 比 Flake8 快 10-100 倍

---

## 安装

```bash
pip install ruff

# 或使用 uv
uv pip install ruff
```

---

## 基本使用

```bash
# 检查代码
ruff check .
ruff check src/

# 自动修复
ruff check --fix .

# 显示修复差异
ruff check --diff .

# 检查特定文件
ruff check path/to/file.py
```

---

## 配置

### pyproject.toml 配置

```toml
[tool.ruff]
# Python 版本
target-version = "py312"

# 行长度
line-length = 88

# 排除目录
exclude = [
    ".git",
    ".venv",
    "__pycache__",
]

[tool.ruff.lint]
# 选择规则
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
]

# 忽略规则
ignore = ["E501"]

# 可自动修复的规则
fixable = ["ALL"]
```

---

## 常用规则集

| 代码 | 名称 | 说明 |
|------|------|------|
| `E` | pycodestyle errors | PEP 8 错误 |
| `W` | pycodestyle warnings | PEP 8 警告 |
| `F` | Pyflakes | 逻辑错误 |
| `I` | isort | import 排序 |
| `UP` | pyupgrade | 代码现代化 |
| `B` | flake8-bugbear | 常见 bug |
| `C4` | flake8-comprehensions | 推导式优化 |
| `SIM` | flake8-simplify | 代码简化 |
| `RUF` | Ruff-specific | Ruff 特有规则 |
| `D` | pydocstyle | 文档检查 |
| `ANN` | flake8-annotations | 类型注解 |

### 推荐配置

```toml
[tool.ruff.lint]
# 基础配置
select = ["E", "F", "I", "W", "UP", "B"]

# 严格配置
select = ["ALL"]
ignore = [
    "D",       # 文档（按需开启）
    "ANN101",  # self 类型注解
    "ANN102",  # cls 类型注解
]
```

---

## import 排序

Ruff 内置 isort 功能：

```toml
[tool.ruff.lint]
select = ["I"]

[tool.ruff.lint.isort]
# 已知的第一方包
known-first-party = ["mypackage"]

# 已知的第三方包
known-third-party = ["requests", "fastapi"]

# 分组
section-order = [
    "future",
    "standard-library",
    "third-party",
    "first-party",
    "local-folder",
]
```

### 排序效果

```python
# 排序前
from mypackage import utils
import os
import requests
from pathlib import Path

# 排序后
import os
from pathlib import Path

import requests

from mypackage import utils
```

---

## 按文件忽略规则

```toml
[tool.ruff.lint.per-file-ignores]
# 测试文件允许 assert
"tests/*" = ["S101"]

# __init__.py 允许未使用的 import
"__init__.py" = ["F401"]

# 迁移文件忽略所有
"migrations/*" = ["ALL"]
```

---

## 行内忽略

```python
# 忽略单行
x = 1  # noqa: E501

# 忽略多个规则
x = 1  # noqa: E501, F841

# 忽略整个文件（文件开头）
# ruff: noqa

# 忽略特定规则整个文件
# ruff: noqa: E501
```

---

## 与 ESLint 对比

| 功能 | Ruff | ESLint |
|------|------|--------|
| 配置文件 | pyproject.toml | .eslintrc |
| 规则选择 | select/ignore | rules |
| 自动修复 | `--fix` | `--fix` |
| 忽略注释 | `# noqa` | `// eslint-disable` |
| 速度 | 极快 | 较慢 |

---

## 常见问题修复

### E501: 行太长

```python
# 问题
result = some_function(argument1, argument2, argument3, argument4, argument5)

# 修复
result = some_function(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5,
)
```

### F401: 未使用的 import

```python
# 问题
import os  # 未使用

# 修复：删除或使用
import os
print(os.getcwd())
```

### I001: import 未排序

```bash
# 自动修复
ruff check --fix --select I .
```

---

## VS Code 集成

```json
// .vscode/settings.json
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
```

---

## 本节要点

1. **Ruff** 是超快的 Python linter
2. 可替代 Flake8 + isort + pyupgrade 等
3. **select** 选择规则，**ignore** 忽略规则
4. **--fix** 自动修复问题
5. 支持 **per-file-ignores** 按文件配置
6. 使用 **# noqa** 忽略特定行

