# 07. 面试题

## 1. ruff 和 black 的区别？

**答案**：

| 特性 | Ruff | Black |
|------|------|-------|
| 功能 | Linter + Formatter | 只有 Formatter |
| 实现 | Rust | Python |
| 速度 | 极快 | 快 |
| 规则 | 可选择 | 固定 |
| 兼容性 | 与 Black 99% 兼容 | 标准 |

**Ruff** 是 linter（代码检查）+ formatter（格式化），可以替代 Flake8 + isort + Black。

**Black** 只是 formatter，专注于代码格式化，且故意提供很少的配置选项。

**推荐**：使用 Ruff 替代 Black + Flake8。

---

## 2. pyright 和 mypy 的区别？

**答案**：

| 特性 | Pyright | MyPy |
|------|---------|------|
| 实现语言 | TypeScript | Python |
| 速度 | 快 | 慢 |
| 错误信息 | 清晰 | 一般 |
| VS Code | 原生支持 | 需插件 |
| 严格程度 | 较严 | 可配置 |
| 维护者 | Microsoft | Python 社区 |

**推荐**：新项目使用 Pyright，更快且错误信息更好。

---

## 3. pre-commit 的原理？

**答案**：

pre-commit 利用 Git 的钩子机制：

1. `pre-commit install` 在 `.git/hooks/pre-commit` 创建脚本
2. 每次 `git commit` 时自动触发
3. 只检查暂存区的文件（staged files）
4. 任何钩子失败则阻止提交

**工作流程**：
```
git add file.py
git commit -m "message"
    ↓
.git/hooks/pre-commit 执行
    ↓
读取 .pre-commit-config.yaml
    ↓
运行配置的钩子（ruff, black, pyright...）
    ↓
全部通过 → 提交成功
任一失败 → 提交失败
```

---

## 4. 如何在 CI 中集成代码检查？

**答案**：

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install tools
        run: pip install ruff pyright

      - name: Ruff check
        run: ruff check .

      - name: Ruff format check
        run: ruff format --check .

      - name: Pyright
        run: pyright
```

或使用 pre-commit action：
```yaml
- uses: pre-commit/action@v3.0.0
```

---

## 5. Python 的类型检查是强制的吗？

**答案**：

**不是强制的**。Python 使用渐进式类型系统（Gradual Typing）：

- **运行时不检查**：类型注解不影响运行
- **外部工具检查**：需要 pyright/mypy 等工具
- **可以逐步添加**：从无类型逐步添加

```python
# 没有类型注解 - 正常运行
def add(a, b):
    return a + b

# 有类型注解 - 正常运行
def add(a: int, b: int) -> int:
    return a + b

# 类型错误 - 运行时不报错！
add("1", "2")  # 只有 pyright/mypy 会报错
```

**好处**：
- 渐进式迁移
- 向后兼容
- 灵活性

---

## 6. 如何处理第三方库没有类型的问题？

**答案**：

### 方案一：安装类型桩

```bash
pip install types-requests
pip install types-redis
pip install pandas-stubs
```

### 方案二：忽略导入

```toml
# pyproject.toml
[tool.pyright]
reportMissingTypeStubs = false

[tool.mypy]
ignore_missing_imports = true
```

### 方案三：创建桩文件

```python
# stubs/some_lib.pyi
def some_function(arg: str) -> int: ...
```

### 方案四：行内忽略

```python
import untyped_lib  # type: ignore
```

---

## 7. py.typed 是什么？

**答案**：

`py.typed` 是一个空的标记文件，放在包根目录：

```
mypackage/
├── __init__.py
├── module.py
└── py.typed  # 标记文件
```

**作用**：
1. 告诉类型检查器此包提供类型信息
2. 安装后类型信息可用
3. PEP 561 标准

**创建**：
```bash
touch src/mypackage/py.typed
```

**pyproject.toml 包含**：
```toml
[tool.hatch.build.targets.wheel]
include = ["py.typed"]
```

---

## 8. 如何忽略某行的类型检查？

**答案**：

### 通用方式（MyPy 和 Pyright）

```python
x = some_untyped_function()  # type: ignore
```

### 忽略特定错误

```python
x = func()  # type: ignore[arg-type]
```

### Pyright 特有

```python
x = func()  # pyright: ignore
x = func()  # pyright: ignore[reportGeneralTypeIssues]
```

### 忽略整个文件

```python
# type: ignore
# 或
# pyright: ignore
```

### 忽略下一行

```python
# type: ignore
next_line = problematic_code()
```

---

## 9. 如何配置严格的类型检查？

**答案**：

### Pyright

```toml
[tool.pyright]
typeCheckingMode = "strict"

# 或手动配置
reportMissingImports = true
reportMissingTypeStubs = true
reportUnusedImport = true
reportUnusedVariable = true
reportUnusedFunction = true
```

### MyPy

```toml
[tool.mypy]
strict = true

# 或手动配置
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_any_generics = true
warn_return_any = true
```

---

## 10. Ruff 的 select 和 ignore 有什么区别？

**答案**：

- **select**: 启用的规则集
- **ignore**: 在 select 基础上排除的规则

```toml
[tool.ruff.lint]
# 启用这些规则
select = ["E", "F", "I", "W"]

# 从中排除
ignore = ["E501", "W503"]
```

**常用规则集**：
- `E`: pycodestyle errors
- `W`: pycodestyle warnings
- `F`: Pyflakes
- `I`: isort
- `UP`: pyupgrade
- `B`: flake8-bugbear
- `ALL`: 所有规则

**严格配置示例**：
```toml
select = ["ALL"]
ignore = ["D", "ANN101"]  # 忽略文档和 self 类型
```

