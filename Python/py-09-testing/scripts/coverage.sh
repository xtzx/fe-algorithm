#!/bin/bash
# 覆盖率报告脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "========================================"
echo "Testing Lab - 覆盖率报告"
echo "========================================"

# 运行测试并收集覆盖率
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "========================================"
echo "覆盖率报告生成完成!"
echo "========================================"
echo ""
echo "HTML 报告位置: htmlcov/index.html"
echo "打开报告: open htmlcov/index.html"

