#!/bin/bash
# 运行测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Running tests..."

# 运行 pytest
python -m pytest tests/ -v --tb=short

echo
echo "All tests passed!"

