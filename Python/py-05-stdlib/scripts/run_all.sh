#!/bin/bash
# 运行所有示例

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../examples"
PROJECT_DIR="$SCRIPT_DIR/../project"

echo "========================================"
echo "P05 标准库精选 - 运行所有示例"
echo "========================================"

# 运行示例
for file in "$EXAMPLES_DIR"/*.py; do
    if [ -f "$file" ]; then
        echo ""
        echo "----------------------------------------"
        echo "运行: $(basename "$file")"
        echo "----------------------------------------"
        python3 "$file"
        echo ""
    fi
done

# 运行项目测试
echo "========================================"
echo "运行项目测试"
echo "========================================"

if [ -f "$PROJECT_DIR/file_organizer/test_organizer.py" ]; then
    echo ""
    echo "----------------------------------------"
    echo "运行: 文件整理器测试"
    echo "----------------------------------------"
    cd "$PROJECT_DIR/file_organizer" && python3 test_organizer.py
fi

echo ""
echo "========================================"
echo "所有示例运行完成!"
echo "========================================"


