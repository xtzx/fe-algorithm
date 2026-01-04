#!/bin/bash

# 测试脚本

set -e

cd "$(dirname "$0")/.."

echo "=== 运行测试 ==="

if [ "$1" = "--cov" ]; then
    pytest tests/ -v --cov=bookmark_api --cov-report=term-missing --cov-report=html
    echo ""
    echo "覆盖率报告: htmlcov/index.html"
else
    pytest tests/ -v
fi

