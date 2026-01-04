# Code Counter - 代码统计工具

一个功能完整的命令行代码统计工具，支持多种编程语言的行数统计。

## ✨ 功能特性

- 📊 **多语言支持**: 自动识别 Python、JavaScript、TypeScript、Go、Rust 等 20+ 种语言
- 📁 **递归扫描**: 扫描整个目录树
- 🚫 **智能排除**: 支持 .gitignore 风格的排除规则
- 📋 **多种输出**: 表格、JSON、Markdown 格式
- ⚙️ **配置文件**: 支持 TOML 配置文件
- 🧪 **完整测试**: 测试覆盖率 > 80%

## 🚀 快速开始

### 安装

```bash
# 从源码安装
pip install -e .

# 或使用 uv
uv pip install -e .
```

### 基本使用

```bash
# 扫描当前目录
code-counter scan .

# 扫描指定目录
code-counter scan /path/to/project

# 排除目录
code-counter scan . --exclude node_modules --exclude .git

# 输出 JSON 格式
code-counter scan . --format json

# 输出 Markdown 格式
code-counter scan . --format markdown > report.md
```

### 命令帮助

```bash
# 查看帮助
code-counter --help

# 查看子命令帮助
code-counter scan --help
code-counter report --help
code-counter config --help
```

## 📖 命令说明

### scan - 扫描目录

```bash
code-counter scan <path> [options]

选项:
  -e, --exclude <pattern>   排除的文件/目录模式（可多次使用）
  -f, --format <format>     输出格式: table, json, markdown
  -o, --output <file>       输出到文件
  --no-ignore               不读取 .gitignore
  -v, --verbose             详细输出
```

### report - 生成报告

```bash
code-counter report <path> [options]

选项:
  --format <format>         报告格式
  --output <file>           输出文件
```

### config - 配置管理

```bash
# 显示当前配置
code-counter config show

# 初始化配置文件
code-counter config init

# 设置配置项
code-counter config set default_format json
```

## ⚙️ 配置文件

创建 `.code-counter.toml` 配置文件：

```toml
# 默认排除的目录
exclude = [
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "dist",
    "build",
]

# 默认输出格式
default_format = "table"

# 语言扩展名映射（自定义）
[languages]
".py" = "Python"
".js" = "JavaScript"
".ts" = "TypeScript"
```

## 📊 输出示例

### 表格格式（默认）

```
╭───────────────┬────────┬────────┬──────────┬───────╮
│ Language      │ Files  │ Code   │ Comments │ Blank │
├───────────────┼────────┼────────┼──────────┼───────┤
│ Python        │ 15     │ 1,234  │ 456      │ 234   │
│ JavaScript    │ 8      │ 567    │ 89       │ 45    │
│ TypeScript    │ 12     │ 890    │ 123      │ 67    │
├───────────────┼────────┼────────┼──────────┼───────┤
│ Total         │ 35     │ 2,691  │ 668      │ 346   │
╰───────────────┴────────┴────────┴──────────┴───────╯
```

### JSON 格式

```json
{
  "summary": {
    "total_files": 35,
    "total_lines": 3705,
    "code_lines": 2691,
    "comment_lines": 668,
    "blank_lines": 346
  },
  "by_language": {
    "Python": { "files": 15, "code": 1234, "comments": 456, "blank": 234 }
  }
}
```

## 🏗️ 项目结构

```
py-11-project-cli/
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── src/code_counter/
│   ├── __init__.py          # 包入口
│   ├── __main__.py          # python -m 入口
│   ├── cli.py               # CLI 定义
│   ├── scanner.py           # 文件扫描器
│   ├── counter.py           # 行数统计器
│   ├── config.py            # 配置管理
│   ├── output.py            # 输出格式化
│   └── models.py            # 数据模型
├── tests/
│   ├── conftest.py          # 测试 fixture
│   ├── test_scanner.py
│   ├── test_counter.py
│   ├── test_config.py
│   └── test_cli.py
└── examples/
    └── sample_project/       # 测试用示例项目
```

## 🧪 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 带覆盖率
pytest --cov=src --cov-report=term-missing

# 代码检查
ruff check src tests
ruff format src tests

# 类型检查
pyright src
```

## 📝 技术栈

- **CLI**: argparse
- **配置**: tomllib (Python 3.11+)
- **文件操作**: pathlib
- **数据模型**: dataclasses
- **类型检查**: pyright
- **代码规范**: ruff
- **测试**: pytest + pytest-cov

## 📄 License

MIT

