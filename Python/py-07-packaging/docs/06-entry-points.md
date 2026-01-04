# 06. Entry Points（CLI 入口）

## 本节目标

- 理解 entry points 机制
- 创建命令行工具
- 对比 package.json 的 bin

---

## 什么是 Entry Points

Entry points 允许包注册可调用对象，最常用于创建 CLI 命令。

```
类比 Node.js:
Python entry_points ≈ package.json bin
```

---

## 创建 CLI 命令

### 项目结构

```
myproject/
├── pyproject.toml
└── src/
    └── mypackage/
        ├── __init__.py
        └── cli.py
```

### cli.py

```python
# src/mypackage/cli.py
import argparse

def main():
    parser = argparse.ArgumentParser(description="My awesome CLI")
    parser.add_argument("name", help="Your name")
    parser.add_argument("-g", "--greeting", default="Hello", help="Greeting")

    args = parser.parse_args()
    print(f"{args.greeting}, {args.name}!")

if __name__ == "__main__":
    main()
```

### pyproject.toml

```toml
[project]
name = "mypackage"
version = "0.1.0"
dependencies = []

[project.scripts]
mycommand = "mypackage.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 安装和使用

```bash
# 安装（开发模式）
pip install -e .

# 使用
mycommand World
# Hello, World!

mycommand World -g "Hi"
# Hi, World!
```

---

## 对比 package.json

### Python (pyproject.toml)

```toml
[project.scripts]
mycli = "mypackage.cli:main"
```

### Node.js (package.json)

```json
{
  "bin": {
    "mycli": "./bin/cli.js"
  }
}
```

### 区别

| 特性 | Python | Node.js |
|------|--------|---------|
| 指向 | 函数 `module:function` | 文件 |
| 执行 | Python 解释器 | Node/直接执行 |
| 安装位置 | venv/bin 或 ~/.local/bin | node_modules/.bin |

---

## entry_points 类型

### console_scripts（CLI 命令）

```toml
[project.scripts]
mycli = "mypackage.cli:main"
another = "mypackage.other:run"
```

安装后：
```bash
which mycli
# /path/to/.venv/bin/mycli
```

### gui_scripts（GUI 应用）

```toml
[project.gui-scripts]
myguiapp = "mypackage.gui:main"
```

在 Windows 上不会弹出控制台窗口。

### 插件入口

```toml
[project.entry-points."myapp.plugins"]
plugin1 = "mypackage.plugins:Plugin1"
plugin2 = "mypackage.plugins:Plugin2"
```

用于插件系统，其他包可以注册插件。

---

## 多命令 CLI

### 使用子命令

```python
# src/mypackage/cli.py
import argparse

def cmd_init(args):
    print(f"Initializing {args.name}")

def cmd_build(args):
    print(f"Building with optimization={args.optimize}")

def main():
    parser = argparse.ArgumentParser(prog="myapp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init 子命令
    init_parser = subparsers.add_parser("init", help="Initialize project")
    init_parser.add_argument("name", help="Project name")
    init_parser.set_defaults(func=cmd_init)

    # build 子命令
    build_parser = subparsers.add_parser("build", help="Build project")
    build_parser.add_argument("-O", "--optimize", action="store_true")
    build_parser.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

使用：
```bash
myapp init myproject
myapp build -O
```

---

## 使用 click（推荐）

click 是更好的 CLI 框架：

```python
# src/mypackage/cli.py
import click

@click.group()
def cli():
    """My awesome CLI tool"""
    pass

@cli.command()
@click.argument("name")
def init(name):
    """Initialize a new project"""
    click.echo(f"Initializing {name}")

@cli.command()
@click.option("-O", "--optimize", is_flag=True, help="Enable optimization")
def build(optimize):
    """Build the project"""
    click.echo(f"Building with optimization={optimize}")

if __name__ == "__main__":
    cli()
```

```toml
[project.scripts]
myapp = "mypackage.cli:cli"

[project.dependencies]
click = ">=8.0"
```

---

## 使用 typer（基于类型提示）

```python
# src/mypackage/cli.py
import typer

app = typer.Typer()

@app.command()
def init(name: str):
    """Initialize a new project"""
    print(f"Initializing {name}")

@app.command()
def build(optimize: bool = typer.Option(False, "-O", "--optimize")):
    """Build the project"""
    print(f"Building with optimization={optimize}")

if __name__ == "__main__":
    app()
```

```toml
[project.scripts]
myapp = "mypackage.cli:app"

[project.dependencies]
typer = ">=0.9"
```

---

## 发现入口点

```python
# 查找已安装包的入口点
from importlib.metadata import entry_points

# Python 3.10+
eps = entry_points(group='console_scripts')
for ep in eps:
    print(f"{ep.name} = {ep.value}")

# 加载入口点
ep = entry_points(group='console_scripts', name='pip')
func = ep[0].load()
```

---

## 调试 CLI

### 直接运行模块

```bash
# 不需要安装
python -m mypackage.cli --help
```

### 开发模式安装

```bash
pip install -e .
mycommand --help
```

### 查看安装的脚本

```bash
ls .venv/bin/ | grep mycommand
```

---

## 本节要点

1. **project.scripts** 定义 CLI 入口
2. 格式：`command = "module:function"`
3. 使用 **click** 或 **typer** 构建复杂 CLI
4. **editable install** 后命令立即可用
5. 类似 package.json 的 bin 字段

