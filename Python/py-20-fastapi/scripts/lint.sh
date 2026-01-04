#!/bin/bash

# 代码质量检查脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 代码质量检查 ===${NC}"

cd "$(dirname "$0")/.."

# Ruff 检查
echo -e "${GREEN}1. Ruff 检查...${NC}"
python -m ruff check src/ tests/

# Ruff 格式检查
echo -e "${GREEN}2. Ruff 格式检查...${NC}"
python -m ruff format --check src/ tests/

# Pyright 类型检查
echo -e "${GREEN}3. Pyright 类型检查...${NC}"
python -m pyright src/

echo ""
echo -e "${GREEN}✅ 所有检查通过!${NC}"

