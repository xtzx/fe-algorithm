#!/bin/bash
# 演示 sample_bad.py 的问题

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

echo "========================================"
echo "演示 sample_bad.py 的问题"
echo "========================================"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo ""
echo "sample_bad.py 故意包含了 20+ 个代码问题"
echo "让我们用 ruff 检查它..."
echo ""
echo ">>> ruff check src/quality_demo/sample_bad.py"
echo "----------------------------------------"

# 临时移除 per-file-ignores 来显示所有问题
ruff check src/quality_demo/sample_bad.py \
    --select=ALL \
    --ignore=D100,D101,D102,D103,D107 \
    2>&1 || true

echo ""
echo ">>> 问题类型说明"
echo "----------------------------------------"
echo "F401: 未使用的 import"
echo "F841: 未使用的变量"
echo "E501: 行太长"
echo "E711: 应该用 'is None' 而不是 '== None'"
echo "E712: 应该用 'if x:' 而不是 'if x == True'"
echo "B006: 可变默认参数"
echo "C400: 不必要的列表推导式"
echo "..."
echo ""
echo "可以用 'ruff check --fix' 自动修复部分问题"
echo "========================================"

