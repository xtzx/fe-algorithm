#!/bin/bash
# 性能分析脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../examples"

cd "$EXAMPLES_DIR"

echo "========================================"
echo "性能分析"
echo "========================================"

if [ -z "$1" ]; then
    echo "用法: $0 <script.py>"
    echo ""
    echo "示例:"
    echo "  $0 profile_demo.py"
    echo "  $0 ../your_script.py"
    exit 1
fi

SCRIPT="$1"

echo ""
echo "1. cProfile (按累计时间)"
echo "----------------------------------------"
python -m cProfile -s cumtime "$SCRIPT" | head -30

echo ""
echo "2. 生成统计文件"
echo "----------------------------------------"
python -m cProfile -o profile.stats "$SCRIPT"
echo "统计文件已生成: profile.stats"

echo ""
echo "提示:"
echo "  - 使用 snakeviz 可视化: snakeviz profile.stats"
echo "  - 安装 snakeviz: pip install snakeviz"

