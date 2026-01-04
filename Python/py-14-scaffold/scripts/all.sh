#!/bin/bash
# 运行所有检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Running all checks..."
echo "=================================================="

# 1. 代码检查
"$SCRIPT_DIR/lint.sh"

# 2. 类型检查
echo ""
"$SCRIPT_DIR/typecheck.sh"

# 3. 测试
echo ""
"$SCRIPT_DIR/test.sh"

echo ""
echo "=================================================="
echo "All checks passed! ✓"
echo "=================================================="

