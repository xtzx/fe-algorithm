# 02. 代码格式化 - Black / Ruff Format

## 本节目标

- 理解代码格式化的重要性
- 掌握 Black 和 Ruff Format 的使用
- 配置团队统一风格

---

## 为什么需要格式化

```
类比 JavaScript:
Black / Ruff Format ≈ Prettier
```

**好处**：
- 消除风格争论
- 代码一致性
- 减少 review 负担
- 自动化处理

---

## Black - "不妥协"的格式化

Black 被称为"不妥协的代码格式化器"——配置选项极少。

### 安装和使用

```bash
# 安装
pip install black

# 格式化
black .
black src/

# 检查但不修改
black --check .

# 显示差异
black --diff .
```

### 配置

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ["py312"]
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | __pycache__
)/
'''
```

**注意**: Black 故意提供很少的配置选项，保持统一风格。

---

## Ruff Format - 更快的替代

Ruff 内置格式化功能，与 Black 99% 兼容但更快。

### 使用

```bash
# 格式化
ruff format .
ruff format src/

# 检查但不修改
ruff format --check .

# 显示差异
ruff format --diff .
```

### 配置

```toml
# pyproject.toml
[tool.ruff.format]
# 引号风格
quote-style = "double"

# 缩进风格
indent-style = "space"

# 跳过魔术尾随逗号
skip-magic-trailing-comma = false

# 行尾
line-ending = "auto"

# 文档字符串格式化
docstring-code-format = true
```

---

## Black vs Ruff Format

| 特性 | Black | Ruff Format |
|------|-------|-------------|
| 速度 | 快 | 极快 (10-100x) |
| 兼容性 | 标准 | 99% Black 兼容 |
| 配置 | 极少 | 稍多 |
| 实现 | Python | Rust |
| 推荐 | ✓ | ✓✓（更推荐） |

### 迁移

从 Black 迁移到 Ruff Format：

```bash
# 几乎无需改动
# black . → ruff format .
# black --check . → ruff format --check .
```

---

## 格式化示例

### 长行处理

```python
# 格式化前
result = some_function(argument1, argument2, argument3, argument4, argument5, argument6)

# 格式化后
result = some_function(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5,
    argument6,
)
```

### 字符串引号

```python
# 格式化前
s = 'hello'
t = "world"

# 格式化后（统一双引号）
s = "hello"
t = "world"
```

### 尾随逗号

```python
# 格式化会添加/保持尾随逗号
my_list = [
    1,
    2,
    3,  # 尾随逗号
]
```

---

## 与 Prettier 对比

| 概念 | Black/Ruff | Prettier |
|------|------------|----------|
| 行长度 | `line-length` | `printWidth` |
| 引号 | `quote-style` | `singleQuote` |
| 尾随逗号 | 自动处理 | `trailingComma` |
| 配置文件 | pyproject.toml | .prettierrc |

---

## 忽略格式化

### 行内忽略

```python
# fmt: off
matrix = [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
]
# fmt: on
```

### 单行忽略

```python
# Black
x = 1  # fmt: skip

# Ruff（也支持）
x = 1  # fmt: skip
```

### 文件忽略

```toml
# pyproject.toml
[tool.black]
exclude = '''
/(
    generated/
)/
'''

[tool.ruff.format]
exclude = ["generated/"]
```

---

## 团队协作

### 1. 统一配置

```toml
# pyproject.toml 提交到版本控制
[tool.ruff.format]
quote-style = "double"
line-length = 88
```

### 2. 编辑器设置

```json
// .vscode/settings.json
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

### 3. Pre-commit 钩子

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff-format
```

### 4. CI 检查

```yaml
# .github/workflows/lint.yml
- name: Check formatting
  run: ruff format --check .
```

---

## VS Code 配置

```json
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.format.args": [],
    "ruff.lint.args": []
}
```

---

## 本节要点

1. **Black** 是"不妥协"的格式化器
2. **Ruff Format** 更快，与 Black 兼容
3. 配置选项故意很少，保持统一
4. 使用 `# fmt: off/on` 忽略格式化
5. 配合 pre-commit 和 CI 自动化
6. 团队统一配置，消除风格争论

