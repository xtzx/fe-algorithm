#!/bin/bash

# 运行测试

set -e

cd "$(dirname "$0")/.."

echo "=== 运行测试 ==="

if [ "$1" = "--cov" ]; then
    pytest tests/ -v --cov=ai_safety --cov-report=term-missing
else
    pytest tests/ -v
fi


