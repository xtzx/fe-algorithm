#!/bin/bash
# 测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "Running tests..."
echo "=================================================="

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# 运行 pytest
pytest tests/ -v --tb=short "$@"

echo ""
echo "=================================================="
echo "All tests passed!"
echo "=================================================="

