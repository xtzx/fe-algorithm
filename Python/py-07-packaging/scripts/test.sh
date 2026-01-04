#!/bin/bash
# 测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "Packaging Lab - 运行测试"
echo "========================================"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 运行测试
pytest tests/ -v --tb=short

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"

