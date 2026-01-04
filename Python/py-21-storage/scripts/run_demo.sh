#!/bin/bash

# 运行演示脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 存储与缓存演示 ===${NC}"

cd "$(dirname "$0")/.."

# 1. 初始化数据库
echo -e "\n${GREEN}1. 初始化数据库...${NC}"
python -m storage_lab.cli db init

# 2. 运行演示
echo -e "\n${GREEN}2. 运行演示...${NC}"
python -m storage_lab.cli demo

# 3. 检查 Redis（可选）
echo -e "\n${GREEN}3. 检查 Redis 连接...${NC}"
python -m storage_lab.cli cache ping || echo "Redis 未连接（可选）"

echo -e "\n${GREEN}✅ 演示完成!${NC}"


