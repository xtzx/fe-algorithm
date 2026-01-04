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
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ZIPAPP_DIR="$PROJECT_DIR/examples/zipapp_demo"

echo -e "${BLUE}=== ZipApp 构建 ===${NC}"

cd "$ZIPAPP_DIR"

# 运行 build.sh
if [ -f "build.sh" ]; then
    chmod +x build.sh
    ./build.sh
else
    echo -e "${GREEN}手动构建...${NC}"
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    cp __main__.py app.py "$TEMP_DIR/"
    
    # 构建
    python -m zipapp "$TEMP_DIR" -o myapp.pyz -p "/usr/bin/env python3"
    
    # 清理
    rm -rf "$TEMP_DIR"
    
    echo -e "\n${GREEN}✅ 构建完成: myapp.pyz${NC}"
fi

