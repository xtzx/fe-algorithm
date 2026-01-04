#!/bin/bash
# 代码检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Code Counter - 代码检查"
echo "========================================"

echo ""
echo "1. Ruff 检查"
echo "----------------------------------------"
ruff check src tests

echo ""
echo "2. Ruff 格式检查"
echo "----------------------------------------"
ruff format --check src tests

echo ""
echo "3. Pyright 类型检查"
echo "----------------------------------------"
pyright src

echo ""
echo "========================================"
echo "所有检查通过!"
echo "========================================"

