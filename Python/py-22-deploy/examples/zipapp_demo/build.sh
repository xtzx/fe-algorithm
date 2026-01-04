#!/bin/bash

# ============================================
# ZipApp 构建脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_NAME="myapp.pyz"

echo -e "${BLUE}=== ZipApp 构建 ===${NC}"

cd "$SCRIPT_DIR"

# 方式 1: 使用 zipapp 模块
echo -e "\n${GREEN}1. 使用 zipapp 构建...${NC}"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cp __main__.py app.py "$TEMP_DIR/"

# 构建 zipapp
python -m zipapp "$TEMP_DIR" -o "$OUTPUT_NAME" -p "/usr/bin/env python3"

# 清理
rm -rf "$TEMP_DIR"

echo -e "\n${GREEN}✅ 构建完成: $OUTPUT_NAME${NC}"

# 测试运行
echo -e "\n${GREEN}2. 测试运行...${NC}"
echo "$ python $OUTPUT_NAME hello --name World"
python "$OUTPUT_NAME" hello --name World

echo ""
echo "$ python $OUTPUT_NAME info"
python "$OUTPUT_NAME" info

echo ""
echo "$ python $OUTPUT_NAME calc '2 + 3 * 4'"
python "$OUTPUT_NAME" calc '2 + 3 * 4'

echo -e "\n${GREEN}✅ 测试通过!${NC}"

# 显示文件大小
echo -e "\n文件大小:"
ls -lh "$OUTPUT_NAME"


