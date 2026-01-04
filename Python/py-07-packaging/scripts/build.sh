#!/bin/bash
# 构建脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Packaging Lab - 构建"
echo "========================================"

# 清理旧的构建产物
if [ -d "dist" ]; then
    echo "清理旧构建..."
    rm -rf dist/
fi

if [ -d "build" ]; then
    rm -rf build/
fi

# 安装 build 工具
pip install build --quiet

# 构建
echo "构建 wheel 和 sdist..."
python -m build

echo ""
echo "========================================"
echo "构建完成!"
echo "========================================"
echo ""
echo "构建产物:"
ls -la dist/
echo ""
echo "安装测试:"
echo "  pip install dist/*.whl"
echo ""

