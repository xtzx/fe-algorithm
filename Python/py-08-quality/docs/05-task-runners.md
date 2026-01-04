# 05. 任务编排 - Make / Just / Nox

## 本节目标

- 理解任务编排的作用
- 掌握常用任务运行器
- 对比 npm scripts

---

## 为什么需要任务编排

```
类比 JavaScript:
make / just / nox ≈ npm scripts
```

Python 没有内置的脚本系统（不像 npm scripts），需要外部工具。

---

## Make - 传统方案

Make 是最传统的任务编排工具。

### Makefile 示例

```makefile
# Makefile
.PHONY: install test lint format clean

# 默认任务
all: lint test

# 安装依赖
install:
	pip install -e ".[dev]"

# 运行测试
test:
	pytest tests/ -v

# 代码检查
lint:
	ruff check src/
	pyright src/

# 格式化
format:
	ruff format src/
	ruff check --fix src/

# 清理
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +

# 构建
build:
	python -m build

# 开发服务器
dev:
	uvicorn app:app --reload
```

### 使用

```bash
make install
make test
make lint
make format
```

### 优缺点

**优点**：
- 几乎所有系统都有
- 简单直观
- 依赖追踪

**缺点**：
- 语法古老
- Windows 支持差
- Tab 敏感

---

## Just - 现代替代

Just 是更现代的命令运行器，专注于运行命令。

### 安装

```bash
# macOS
brew install just

# 其他
cargo install just
```

### justfile 示例

```just
# justfile
# 默认任务
default: lint test

# 安装依赖
install:
    pip install -e ".[dev]"

# 运行测试
test *args:
    pytest tests/ {{args}}

# 代码检查
lint:
    ruff check src/
    pyright src/

# 格式化
format:
    ruff format src/
    ruff check --fix src/

# 清理
clean:
    rm -rf build/ dist/ *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} +

# 带参数的任务
greet name="World":
    echo "Hello, {{name}}!"

# 设置环境变量
set dotenv-load := true

# 条件执行
ci: lint test build
```

### 使用

```bash
just install
just test
just test -v --tb=short
just greet "Python"
```

### Just vs Make

| 特性 | Just | Make |
|------|------|------|
| 语法 | 简单 | 复杂 |
| 参数 | 原生支持 | 复杂 |
| 跨平台 | 好 | Windows 差 |
| 安装 | 需要安装 | 内置 |

---

## Nox - Python 原生

Nox 用 Python 编写任务，类似 tox 但更灵活。

### 安装

```bash
pip install nox
```

### noxfile.py 示例

```python
# noxfile.py
import nox

# 默认会话
nox.options.sessions = ["lint", "test"]

@nox.session
def lint(session):
    """运行代码检查"""
    session.install("ruff", "pyright")
    session.run("ruff", "check", "src/")
    session.run("pyright", "src/")

@nox.session
def format(session):
    """格式化代码"""
    session.install("ruff")
    session.run("ruff", "format", "src/")
    session.run("ruff", "check", "--fix", "src/")

@nox.session(python=["3.10", "3.11", "3.12"])
def test(session):
    """运行测试（多 Python 版本）"""
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/", "-v")

@nox.session
def build(session):
    """构建包"""
    session.install("build")
    session.run("python", "-m", "build")

@nox.session
def docs(session):
    """构建文档"""
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build")
```

### 使用

```bash
# 运行默认会话
nox

# 运行特定会话
nox -s lint
nox -s test

# 列出所有会话
nox -l

# 多 Python 版本测试
nox -s test
```

### Nox 优势

- Python 语法
- 多 Python 版本测试
- 自动创建虚拟环境
- 灵活的参数化

---

## Tox - 多环境测试

Tox 专注于多环境测试。

### tox.ini 示例

```ini
# tox.ini
[tox]
envlist = py310,py311,py312,lint

[testenv]
deps = pytest
commands = pytest tests/

[testenv:lint]
deps =
    ruff
    pyright
commands =
    ruff check src/
    pyright src/

[testenv:format]
deps = ruff
commands = ruff format src/
```

### 使用

```bash
tox
tox -e lint
tox -e py312
```

---

## 工具对比

| 工具 | 语言 | 多 Python 版本 | 复杂度 |
|------|------|----------------|--------|
| Make | Makefile | 不支持 | 简单 |
| Just | Justfile | 不支持 | 简单 |
| Nox | Python | 支持 | 中等 |
| Tox | INI | 支持 | 中等 |
| npm scripts | JSON | N/A | 简单 |

### 选择建议

| 场景 | 推荐 |
|------|------|
| 简单项目 | Make 或 Just |
| 跨平台 | Just |
| 多 Python 版本 | Nox 或 Tox |
| 熟悉 Python | Nox |
| CI/CD | Nox |

---

## 与 npm scripts 对比

**package.json:**
```json
{
  "scripts": {
    "test": "jest",
    "lint": "eslint src/",
    "format": "prettier --write src/",
    "build": "tsc"
  }
}
```

**Makefile:**
```makefile
test:
	pytest
lint:
	ruff check src/
format:
	ruff format src/
build:
	python -m build
```

---

## 推荐配置

### 简单项目：Makefile

```makefile
.PHONY: all install test lint format clean

all: lint test

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && pyright

format:
	ruff format . && ruff check --fix .

clean:
	rm -rf build/ dist/ __pycache__/
```

### 复杂项目：noxfile.py

```python
import nox

@nox.session(python=["3.10", "3.11", "3.12"])
def test(session):
    session.install("-e", ".[dev]")
    session.run("pytest")

@nox.session
def lint(session):
    session.install("ruff", "pyright")
    session.run("ruff", "check", ".")
    session.run("pyright")
```

---

## 本节要点

1. **Make** 是传统方案，广泛可用
2. **Just** 更现代，语法简单
3. **Nox** 用 Python 写，支持多版本
4. **Tox** 专注多环境测试
5. 根据项目复杂度选择工具

