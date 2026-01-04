#!/bin/bash
# 数据处理演示脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
DATA_DIR="$PROJECT_DIR/data"

cd "$PROJECT_DIR"

echo "========================================"
echo "Data Lab - 数据处理演示"
echo "========================================"

# 安装依赖
echo ""
echo "1. 安装依赖"
echo "----------------------------------------"
pip install -e . -q

echo ""
echo "2. 查看脏数据"
echo "----------------------------------------"
echo "文件: data/dirty.csv"
head -5 "$DATA_DIR/dirty.csv"
echo "..."

echo ""
echo "3. 生成数据质量报告"
echo "----------------------------------------"
python -m data_lab report "$DATA_DIR/dirty.csv"

echo ""
echo "4. 清洗数据"
echo "----------------------------------------"
python -m data_lab clean "$DATA_DIR/dirty.csv" \
    -o "$DATA_DIR/clean.jsonl" \
    --report "$DATA_DIR/report.json"

echo ""
echo "5. 查看清洗后的数据"
echo "----------------------------------------"
echo "文件: data/clean.jsonl"
head -5 "$DATA_DIR/clean.jsonl"

echo ""
echo "6. 验证清洗后的数据"
echo "----------------------------------------"
python -m data_lab validate "$DATA_DIR/clean.jsonl"

echo ""
echo "========================================"
echo "演示完成!"
echo "========================================"
echo ""
echo "生成的文件:"
echo "  - data/clean.jsonl (清洗后的数据)"
echo "  - data/report.json (清洗报告)"

