#!/bin/bash
# 运行所有 P03 示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$PROJECT_DIR/examples"

echo "========================================"
echo "P03: 面向对象编程 - 运行所有示例"
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
echo "运行扑克牌游戏项目"
echo "========================================"
cd "$PROJECT_DIR/project/poker_game"
python3 main.py

echo ""
echo "========================================"
echo "运行测试"
echo "========================================"
python3 -m pytest test_poker.py -v 2>/dev/null || python3 test_poker.py

echo ""
echo "========================================"
echo "✅ 所有示例和测试运行完成"
echo "========================================"

