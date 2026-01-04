#!/bin/bash
# 测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Data Lab - 运行测试"
echo "========================================"

# 安装依赖
pip install -e ".[dev]" -q

# 运行测试
pytest tests/ -v --cov=src --cov-report=term-missing

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"

