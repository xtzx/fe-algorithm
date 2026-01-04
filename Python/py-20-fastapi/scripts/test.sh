#!/bin/bash

# 测试脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 运行测试 ===${NC}"

cd "$(dirname "$0")/.."

# 运行测试
if [ "$1" = "--cov" ]; then
    echo -e "${GREEN}运行测试（带覆盖率）...${NC}"
    python -m pytest tests/ -v --cov=api --cov-report=term-missing --cov-report=html
    echo ""
    echo "覆盖率报告: htmlcov/index.html"
else
    echo -e "${GREEN}运行测试...${NC}"
    python -m pytest tests/ -v
fi

