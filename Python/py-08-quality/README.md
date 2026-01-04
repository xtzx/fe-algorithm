# P08: 工程质量工具链

> 等价于 ESLint/Prettier/TypeScript 的 Python 方案

## 学完后能做

- 配置 ruff/black/pyright
- 使用 pre-commit 自动化检查
- 保持代码质量

## 快速开始

```bash
# 安装工具
pip install ruff black pyright pre-commit

# 运行检查
ruff check src/
ruff format src/
pyright src/

# 设置 pre-commit
pre-commit install
```

## Python vs JavaScript 工具对比

| 功能 | Python | JavaScript |
|------|--------|------------|
| Linter | ruff | ESLint |
| Formatter | black / ruff format | Prettier |
| Type Checker | pyright / mypy | TypeScript |
| Git Hooks | pre-commit | husky + lint-staged |
| Task Runner | make / just / nox | npm scripts |

## 目录结构

```
py-08-quality/
├── README.md
├── pyproject.toml              # 工具配置
├── .pre-commit-config.yaml     # pre-commit 配置
├── docs/
│   ├── 01-ruff.md              # ruff 配置与使用
│   ├── 02-formatter.md         # black/ruff format
│   ├── 03-type-checking.md     # pyright/mypy
│   ├── 04-pre-commit.md        # pre-commit 设置
│   ├── 05-task-runners.md      # make/just/nox
│   ├── 06-exercises.md         # 练习题
│   └── 07-interview-questions.md # 面试题
├── src/quality_demo/
│   ├── sample_good.py          # 良好代码示例
│   └── sample_bad.py           # 问题代码示例
└── scripts/
```

## 工具选择速查

### 推荐配置（2024）

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]

[tool.ruff.format]
quote-style = "double"

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "standard"
```

### 一键检查命令

```bash
# 检查所有
ruff check . && ruff format --check . && pyright

# 自动修复
ruff check --fix . && ruff format .
```

## 常见配置模式

### 严格模式（推荐）

```toml
[tool.ruff.lint]
select = ["ALL"]
ignore = ["D", "ANN101", "ANN102"]

[tool.pyright]
typeCheckingMode = "strict"
```

### 宽松模式（遗留项目）

```toml
[tool.ruff.lint]
select = ["E", "F"]

[tool.pyright]
typeCheckingMode = "basic"
```

## 学习路径

1. [ruff 配置与使用](docs/01-ruff.md)
2. [代码格式化](docs/02-formatter.md)
3. [类型检查](docs/03-type-checking.md)
4. [pre-commit 设置](docs/04-pre-commit.md)
5. [任务编排](docs/05-task-runners.md)

