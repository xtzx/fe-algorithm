#!/bin/bash
# 代码检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "Running code linting..."
echo "=================================================="

# Ruff 检查
echo ""
echo "[1/2] Ruff check..."
ruff check .

# Ruff 格式检查
echo ""
echo "[2/2] Ruff format check..."
ruff format --check .

echo ""
echo "=================================================="
echo "All linting checks passed!"
echo "=================================================="

