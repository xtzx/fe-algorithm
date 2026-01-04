#!/bin/bash
# 代码格式化脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "Formatting code..."
echo "=================================================="

# Ruff 格式化
echo ""
echo "[1/2] Ruff format..."
ruff format .

# Ruff 自动修复
echo ""
echo "[2/2] Ruff check --fix..."
ruff check --fix .

echo ""
echo "=================================================="
echo "Code formatting completed!"
echo "=================================================="

