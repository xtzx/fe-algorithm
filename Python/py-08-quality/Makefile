# Makefile - 任务编排
# 使用: make <target>

.PHONY: all install lint format check test clean help

# 默认任务
all: lint

# 显示帮助
help:
	@echo "可用命令:"
	@echo "  make install    - 安装依赖"
	@echo "  make lint       - 运行代码检查 (ruff + pyright)"
	@echo "  make format     - 格式化代码"
	@echo "  make check      - 检查格式 (不修改)"
	@echo "  make test       - 运行测试"
	@echo "  make clean      - 清理构建产物"
	@echo "  make pre-commit - 设置 pre-commit"

# 安装依赖
install:
	pip install -e ".[dev]"

# 代码检查
lint:
	@echo ">>> Running ruff check..."
	ruff check src/
	@echo ">>> Running pyright..."
	pyright src/

# 格式化代码
format:
	@echo ">>> Formatting with ruff..."
	ruff format src/
	@echo ">>> Fixing with ruff..."
	ruff check --fix src/

# 检查格式（不修改）
check:
	@echo ">>> Checking format..."
	ruff format --check src/
	@echo ">>> Checking lint..."
	ruff check src/

# 运行测试
test:
	pytest tests/ -v

# 清理
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

# 设置 pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

