#!/bin/bash
# 运行测试覆盖率

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "运行测试覆盖率分析"
echo "=================================================="
echo

# 检查依赖
if ! command -v pytest &> /dev/null; then
    echo "正在安装依赖..."
    pip install pytest pytest-cov
fi

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# 运行测试并生成覆盖率报告
pytest tests/ \
    --cov=file_automation \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    -v

echo
echo "=================================================="
echo "覆盖率报告已生成"
echo "HTML 报告: $PROJECT_DIR/htmlcov/index.html"
echo "=================================================="

