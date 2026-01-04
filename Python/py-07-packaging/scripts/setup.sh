#!/bin/bash
# 项目设置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Packaging Lab - 项目设置"
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

echo ""
echo "========================================"
echo "设置完成!"
echo "========================================"
echo ""
echo "使用以下命令激活环境:"
echo "  source .venv/bin/activate"
echo ""
echo "可用命令:"
echo "  pkglab --help     # 查看 CLI 帮助"
echo "  pytest            # 运行测试"
echo "  ruff check src/   # 代码检查"
echo ""

