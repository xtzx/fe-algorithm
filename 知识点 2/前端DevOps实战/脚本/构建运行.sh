#!/bin/bash

# ============================================
# 本地构建和运行脚本
#
# 用途：本地开发和测试时快速构建运行整个应用
#
# 使用：
#   chmod +x build-and-run.sh
#   ./build-and-run.sh
# ============================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 配置
# ============================================
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DOCKER_COMPOSE_DIR="$PROJECT_ROOT/examples/docker-compose"

# ============================================
# 前置检查
# ============================================
log_info "检查 Docker 是否安装..."
if ! command -v docker &> /dev/null; then
    log_error "Docker 未安装，请先安装 Docker"
    exit 1
fi

log_info "检查 Docker Compose 是否安装..."
if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

log_info "检查 Docker 守护进程是否运行..."
if ! docker info &> /dev/null; then
    log_error "Docker 守护进程未运行，请启动 Docker"
    exit 1
fi

# ============================================
# 停止旧容器
# ============================================
log_info "停止已存在的容器..."
cd "$DOCKER_COMPOSE_DIR"
docker-compose down --remove-orphans || true

# ============================================
# 构建镜像
# ============================================
log_info "构建 Docker 镜像..."
docker-compose build --no-cache

# ============================================
# 启动服务
# ============================================
log_info "启动所有服务..."
docker-compose up -d

# ============================================
# 等待服务就绪
# ============================================
log_info "等待服务启动..."
sleep 5

# 检查服务状态
log_info "检查服务状态..."
docker-compose ps

# ============================================
# 健康检查
# ============================================
log_info "执行健康检查..."

MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f http://localhost/health > /dev/null 2>&1; then
        log_info "健康检查通过 ✓"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_warn "等待服务就绪... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "服务启动失败，请检查日志："
    docker-compose logs --tail=50
    exit 1
fi

# ============================================
# 完成
# ============================================
echo ""
log_info "=========================================="
log_info "  服务启动成功！"
log_info "=========================================="
log_info "  前端: http://localhost"
log_info "  API:  http://localhost/api/visits"
log_info "  健康: http://localhost/health"
log_info ""
log_info "  查看日志: docker-compose logs -f"
log_info "  停止服务: docker-compose down"
log_info "=========================================="

