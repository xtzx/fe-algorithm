#!/bin/bash
# 运行所有示例

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../examples"

echo "========================================"
echo "P06 Python 运行时原理 - 运行所有示例"
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

echo "========================================"
echo "所有示例运行完成!"
echo "========================================"

