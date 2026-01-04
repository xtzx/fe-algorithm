#!/bin/bash

# FastAPI 开发服务器启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== FastAPI 开发服务器 ===${NC}"

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "提示: 建议先激活虚拟环境"
    echo "  source .venv/bin/activate"
fi

# 安装依赖（如果需要）
if [ "$1" = "--install" ]; then
    echo -e "${GREEN}安装依赖...${NC}"
    pip install -e ".[dev]"
fi

# 启动服务器
echo -e "${GREEN}启动 FastAPI 服务器...${NC}"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""

cd "$(dirname "$0")/.."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

