# 03. 类型检查 - Pyright / MyPy

## 本节目标

- 理解 Python 类型检查的意义
- 掌握 Pyright 和 MyPy 的配置
- 处理第三方库类型问题

---

## 类型检查概述

```
类比 JavaScript:
Pyright / MyPy ≈ TypeScript
```

**Python 类型检查是可选的**（渐进式类型系统）：
- 运行时不强制
- 由外部工具检查
- 可以逐步添加

---

## Pyright vs MyPy

| 特性 | Pyright | MyPy |
|------|---------|------|
| 速度 | 快 | 慢 |
| 实现 | TypeScript | Python |
| 严格程度 | 较严 | 可配置 |
| VS Code | 内置支持 | 需插件 |
| 错误信息 | 清晰 | 一般 |
| 推荐 | ✓✓ | ✓ |

**推荐 Pyright**：更快、错误信息更好、VS Code 原生支持。

---

## Pyright 使用

### 安装

```bash
pip install pyright

# 或使用 npm（更常见）
npm install -g pyright
```

### 运行

```bash
# 检查当前目录
pyright

# 检查特定路径
pyright src/

# 查看配置
pyright --verifytypes package_name
```

### 配置

```toml
# pyproject.toml
[tool.pyright]
pythonVersion = "3.12"
pythonPlatform = "All"

# 类型检查模式
# off, basic, standard, strict, all
typeCheckingMode = "standard"

# 包含/排除
include = ["src"]
exclude = [".venv", "**/__pycache__"]

# 报告设置
reportMissingImports = true
reportMissingTypeStubs = false
reportUnusedImport = true
reportUnusedVariable = true
```

---

## MyPy 使用

### 安装

```bash
pip install mypy
```

### 运行

```bash
# 检查
mypy src/

# 严格模式
mypy --strict src/
```

### 配置

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"

# 严格模式
strict = true

# 或单独配置
warn_return_any = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

# 忽略缺少类型的导入
ignore_missing_imports = true

# 排除
exclude = ["tests/"]
```

---

## 类型检查模式

### Pyright 模式

| 模式 | 严格程度 | 适用场景 |
|------|----------|----------|
| off | 无 | 禁用 |
| basic | 低 | 遗留代码 |
| standard | 中 | 推荐 |
| strict | 高 | 新项目 |
| all | 最高 | 完美主义 |

### 推荐配置

```toml
# 新项目
[tool.pyright]
typeCheckingMode = "strict"

# 遗留项目
[tool.pyright]
typeCheckingMode = "basic"
```

---

## 类型注解基础

```python
# 基础类型
def greet(name: str) -> str:
    return f"Hello, {name}"

# 可选类型
def find(items: list[int], target: int) -> int | None:
    for i, item in enumerate(items):
        if item == target:
            return i
    return None

# 复杂类型
from typing import Callable, TypeVar

T = TypeVar("T")

def apply(func: Callable[[T], T], value: T) -> T:
    return func(value)
```

---

## 处理第三方库类型

### 问题：库没有类型

```python
import some_lib  # error: Cannot find type stubs
```

### 解决方案 1：安装类型桩

```bash
# 很多库有对应的类型桩
pip install types-requests
pip install types-redis
pip install pandas-stubs
```

### 解决方案 2：忽略导入

```toml
# pyproject.toml
[tool.pyright]
reportMissingTypeStubs = false

[tool.mypy]
ignore_missing_imports = true
```

### 解决方案 3：创建桩文件

```python
# stubs/some_lib.pyi
def some_function(arg: str) -> int: ...
```

---

## py.typed 标记

`py.typed` 是一个空文件，表示包提供类型信息：

```
mypackage/
├── __init__.py
├── module.py
└── py.typed  # 标记文件
```

**作用**：
- 告诉类型检查器此包有类型
- 安装后类型信息可用

---

## 忽略类型错误

### 行内忽略

```python
# Pyright
x = some_untyped_function()  # type: ignore

# 更具体
x = func()  # type: ignore[arg-type]

# Pyright 特有
x = func()  # pyright: ignore
x = func()  # pyright: ignore[reportGeneralTypeIssues]
```

### 文件忽略

```python
# 文件开头
# pyright: strict
# pyright: ignore
# type: ignore
```

---

## 渐进式类型

### 策略：逐步添加类型

1. **关键路径先行**：核心模块先加类型
2. **新代码必须**：新代码要求类型
3. **逐步收紧**：basic → standard → strict

### 示例配置

```toml
# 基础检查，逐步收紧
[tool.pyright]
typeCheckingMode = "basic"
reportMissingImports = true
reportUnusedVariable = true

# 特定目录更严格
[[tool.pyright.executionEnvironments]]
root = "src/core"
typeCheckingMode = "strict"
```

---

## 与 TypeScript 对比

| 概念 | Python | TypeScript |
|------|--------|------------|
| 类型注解 | `def f(x: int) -> str` | `function f(x: number): string` |
| 联合类型 | `int \| str` | `number \| string` |
| 可选 | `x: int \| None` | `x?: number` |
| 泛型 | `T = TypeVar('T')` | `<T>` |
| 类型文件 | `.pyi` | `.d.ts` |
| 类型检查 | 可选/外部工具 | 编译时强制 |

---

## VS Code 配置

```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "standard",
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.autoImportCompletions": true
}
```

---

## 本节要点

1. **Pyright** 更快、推荐使用
2. **typeCheckingMode** 控制严格程度
3. 使用 **type: ignore** 忽略特定行
4. **py.typed** 标记包提供类型
5. 渐进式添加类型，逐步收紧
6. 类型检查是可选的，但推荐使用

