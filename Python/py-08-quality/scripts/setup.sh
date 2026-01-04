#!/bin/bash
# 项目设置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Quality Demo - 项目设置"
echo "========================================"

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -e ".[dev]"

# 安装 pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "安装 pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "========================================"
echo "设置完成!"
echo "========================================"
echo ""
echo "使用以下命令激活环境:"
echo "  source .venv/bin/activate"
echo ""
echo "可用命令:"
echo "  make lint       # 代码检查"
echo "  make format     # 格式化代码"
echo "  make check      # 检查格式"
echo "  ruff check src/ # 直接运行 ruff"
echo "  pyright src/    # 直接运行 pyright"
echo ""

