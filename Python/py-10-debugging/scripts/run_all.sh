#!/bin/bash
# 运行所有示例

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../examples"

cd "$EXAMPLES_DIR"

echo "========================================"
echo "调试与性能优化 - 运行所有示例"
echo "========================================"

echo ""
echo "1. pdb 调试演示"
echo "----------------------------------------"
# 注意：pdb_demo.py 中有 breakpoint()，会暂停
# 这里跳过，建议单独运行
echo "跳过 (包含 breakpoint，请单独运行)"
echo "运行命令: python pdb_demo.py"

echo ""
echo "2. logging 演示"
echo "----------------------------------------"
python logging_demo.py

echo ""
echo "3. 性能分析演示"
echo "----------------------------------------"
python profile_demo.py

echo ""
echo "4. 内存分析演示"
echo "----------------------------------------"
python memory_demo.py

echo ""
echo "========================================"
echo "所有示例运行完成!"
echo "========================================"

