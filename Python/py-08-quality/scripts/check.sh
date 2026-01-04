#!/bin/bash
# 代码检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Quality Demo - 代码检查"
echo "========================================"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo ""
echo ">>> Ruff Check"
echo "----------------------------------------"
ruff check src/ || true

echo ""
echo ">>> Ruff Format Check"
echo "----------------------------------------"
ruff format --check src/ || true

echo ""
echo ">>> Pyright"
echo "----------------------------------------"
pyright src/ || true

echo ""
echo "========================================"
echo "检查完成!"
echo "========================================"

