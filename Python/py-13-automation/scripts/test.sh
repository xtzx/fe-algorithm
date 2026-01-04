#!/bin/bash
# 运行测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "运行 py-13-automation 测试"
echo "=================================================="
echo

# 检查是否安装了 pytest
if ! command -v pytest &> /dev/null; then
    echo "正在安装 pytest..."
    pip install pytest pytest-cov
fi

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# 运行测试
echo "运行测试..."
pytest tests/ -v --tb=short

echo
echo "=================================================="
echo "测试完成！"
echo "=================================================="

