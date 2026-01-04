#!/bin/bash
# ============================================
# 测试运行脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}运行测试...${NC}"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 设置测试环境
export APP_ENV=testing
export LLM_PROVIDER=stub
export EMBEDDING_PROVIDER=stub

# 运行 pytest
pytest tests/ \
    -v \
    --cov=knowledge_assistant \
    --cov-report=term-missing \
    --cov-report=html:coverage_html \
    "$@"

echo -e "${GREEN}测试完成！${NC}"
echo -e "覆盖率报告: coverage_html/index.html"


