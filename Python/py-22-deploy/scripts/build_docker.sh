#!/bin/bash

# ============================================
# Docker 构建脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 配置
IMAGE_NAME="${IMAGE_NAME:-myapp}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/examples/docker"

echo -e "${BLUE}=== Docker 构建 ===${NC}"

cd "$DOCKER_DIR"

# 检查 Dockerfile
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}错误: $DOCKERFILE 不存在${NC}"
    exit 1
fi

echo -e "\n${GREEN}1. 构建镜像...${NC}"
echo "   镜像名称: $IMAGE_NAME:$IMAGE_TAG"
echo "   Dockerfile: $DOCKERFILE"

docker build \
    -f "$DOCKERFILE" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    .

echo -e "\n${GREEN}2. 镜像信息${NC}"
docker images "$IMAGE_NAME:$IMAGE_TAG"

# 显示镜像大小
SIZE=$(docker images "$IMAGE_NAME:$IMAGE_TAG" --format "{{.Size}}")
echo "   镜像大小: $SIZE"

echo -e "\n${GREEN}3. 测试运行...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 "$IMAGE_NAME:$IMAGE_TAG")
echo "   容器 ID: $CONTAINER_ID"

# 等待启动
sleep 3

# 健康检查
echo -e "\n${GREEN}4. 健康检查...${NC}"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   ✅ 健康检查通过"
else
    echo "   ❌ 健康检查失败"
fi

# 清理测试容器
docker stop "$CONTAINER_ID" > /dev/null
docker rm "$CONTAINER_ID" > /dev/null

echo -e "\n${GREEN}✅ 构建完成!${NC}"
echo ""
echo "运行命令:"
echo "  docker run -d -p 8000:8000 $IMAGE_NAME:$IMAGE_TAG"

