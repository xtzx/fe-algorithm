#!/bin/bash
# 运行测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Code Counter - 运行测试"
echo "========================================"

# 安装开发依赖
echo "安装依赖..."
pip install -e ".[dev]" -q

echo ""
echo "运行测试（带覆盖率）..."
echo "----------------------------------------"
pytest tests/ -v --cov=src --cov-report=term-missing

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"

