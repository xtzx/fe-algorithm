#!/bin/bash
# ============================================
# 开发环境启动脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}启动开发服务器...${NC}"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}建议在虚拟环境中运行${NC}"
fi

# 设置环境变量
export APP_ENV=development
export DEBUG=true
export LOG_LEVEL=DEBUG
export LOG_FORMAT=console
export LLM_PROVIDER=stub
export EMBEDDING_PROVIDER=stub

# 确保数据目录存在
mkdir -p data/uploads data/index data/eval_dataset

# 启动服务
uvicorn knowledge_assistant.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src

echo -e "${GREEN}服务已停止${NC}"


