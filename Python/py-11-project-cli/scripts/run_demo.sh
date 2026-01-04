#!/bin/bash
# 运行演示脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
SAMPLE_PROJECT="$PROJECT_DIR/examples/sample_project"

cd "$PROJECT_DIR"

echo "========================================"
echo "Code Counter - 演示"
echo "========================================"

# 检查是否安装
if ! command -v code-counter &> /dev/null; then
    echo "安装 code-counter..."
    pip install -e .
fi

echo ""
echo "1. 扫描示例项目（表格格式）"
echo "----------------------------------------"
code-counter scan "$SAMPLE_PROJECT"

echo ""
echo "2. 扫描示例项目（JSON 格式）"
echo "----------------------------------------"
code-counter scan "$SAMPLE_PROJECT" -f json

echo ""
echo "3. 扫描示例项目（Markdown 格式）"
echo "----------------------------------------"
code-counter scan "$SAMPLE_PROJECT" -f markdown

echo ""
echo "4. 排除 JavaScript 文件"
echo "----------------------------------------"
code-counter scan "$SAMPLE_PROJECT" -e "*.js"

echo ""
echo "5. 显示配置"
echo "----------------------------------------"
code-counter config show

echo ""
echo "========================================"
echo "演示完成!"
echo "========================================"

