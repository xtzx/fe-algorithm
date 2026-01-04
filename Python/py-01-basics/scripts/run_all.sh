#!/bin/bash
# 运行所有 P01 示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$PROJECT_DIR/examples"

echo "========================================"
echo "P01: Python 基础语法 - 运行所有示例"
echo "========================================"
echo ""

cd "$EXAMPLES_DIR"

for file in *.py; do
    if [ -f "$file" ]; then
        echo "----------------------------------------"
        echo "运行: $file"
        echo "----------------------------------------"
        python3 "$file"
        echo ""
    fi
done

echo "========================================"
echo "运行测试: 文本统计器"
echo "========================================"
cd "$PROJECT_DIR/project/text_analyzer"
python3 -m pytest test_analyzer.py -v 2>/dev/null || python3 test_analyzer.py

echo ""
echo "========================================"
echo "✅ 所有示例运行完成"
echo "========================================"

