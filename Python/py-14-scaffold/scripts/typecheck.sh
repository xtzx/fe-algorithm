#!/bin/bash
# 类型检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "Running type checking..."
echo "=================================================="

# Pyright 类型检查
pyright

echo ""
echo "=================================================="
echo "Type checking passed!"
echo "=================================================="

