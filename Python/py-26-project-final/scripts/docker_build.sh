#!/bin/bash
# ============================================
# Docker 构建脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认参数
IMAGE_NAME="${IMAGE_NAME:-knowledge-assistant}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo -e "${GREEN}构建 Docker 镜像...${NC}"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 构建镜像
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo -e "${GREEN}镜像构建完成: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# 显示镜像信息
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo -e "${YELLOW}使用以下命令启动:${NC}"
echo "  docker-compose up -d"
echo ""
echo -e "${YELLOW}或单独运行:${NC}"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"


