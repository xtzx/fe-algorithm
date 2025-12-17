#!/bin/bash

# ============================================
# 生产环境部署脚本
#
# 用途：在生产服务器上部署应用
#
# 使用：
#   chmod +x deploy-with-docker.sh
#   ./deploy-with-docker.sh [tag]
#
#   例如：
#   ./deploy-with-docker.sh latest
#   ./deploy-with-docker.sh v1.2.3
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 配置（根据实际情况修改）
# ============================================
REGISTRY="ghcr.io"
IMAGE_NAME="your-org/your-app"  # 修改为实际镜像名
TAG="${1:-latest}"              # 默认使用 latest 标签
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

APP_DIR="/opt/app"              # 应用部署目录
BACKUP_DIR="/opt/backups"       # 备份目录

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# ============================================
# 前置检查
# ============================================
log_step "1/7 前置检查..."

if [ ! -d "$APP_DIR" ]; then
    log_error "应用目录不存在: $APP_DIR"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "Docker 未安装"
    exit 1
fi

# ============================================
# 备份当前版本
# ============================================
log_step "2/7 备份当前版本..."

mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/docker-compose-$(date '+%Y%m%d%H%M%S').yml"

if [ -f "$APP_DIR/docker-compose.yml" ]; then
    cp "$APP_DIR/docker-compose.yml" "$BACKUP_FILE"
    log_info "配置已备份到: $BACKUP_FILE"
fi

# 记录当前运行的镜像版本
CURRENT_IMAGE=$(docker-compose -f "$APP_DIR/docker-compose.yml" images -q api 2>/dev/null || echo "none")
log_info "当前镜像: $CURRENT_IMAGE"

# ============================================
# 拉取新镜像
# ============================================
log_step "3/7 拉取新镜像: $FULL_IMAGE"

docker pull "$FULL_IMAGE"
log_info "镜像拉取完成"

# ============================================
# 停止旧服务
# ============================================
log_step "4/7 停止旧服务..."

cd "$APP_DIR"
docker-compose down --remove-orphans
log_info "旧服务已停止"

# ============================================
# 启动新服务
# ============================================
log_step "5/7 启动新服务..."

docker-compose up -d
log_info "新服务已启动"

# ============================================
# 健康检查
# ============================================
log_step "6/7 健康检查..."

MAX_RETRIES=30
RETRY_COUNT=0
HEALTH_URL="http://localhost/health"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")

    if [ "$RESPONSE" = "200" ]; then
        log_info "健康检查通过 ✓"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_warn "等待服务就绪... ($RETRY_COUNT/$MAX_RETRIES) HTTP: $RESPONSE"
        sleep 2
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "健康检查失败，开始回滚..."

    # 回滚
    docker-compose down
    if [ -n "$CURRENT_IMAGE" ] && [ "$CURRENT_IMAGE" != "none" ]; then
        log_warn "回滚到之前版本..."
        docker-compose up -d
    fi

    log_error "部署失败！请检查日志："
    docker-compose logs --tail=50
    exit 1
fi

# ============================================
# 清理
# ============================================
log_step "7/7 清理旧资源..."

# 删除未使用的镜像
docker image prune -f
log_info "旧镜像已清理"

# 删除 30 天前的备份
find "$BACKUP_DIR" -name "*.yml" -mtime +30 -delete 2>/dev/null || true
log_info "旧备份已清理"

# ============================================
# 完成
# ============================================
echo ""
log_info "=========================================="
log_info "  部署成功！"
log_info "=========================================="
log_info "  镜像: $FULL_IMAGE"
log_info "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_info ""
log_info "  服务状态:"
docker-compose ps
log_info "=========================================="

# 可选：发送通知
# curl -X POST "https://your-webhook.com" \
#     -H "Content-Type: application/json" \
#     -d "{\"text\": \"✅ 部署成功: $FULL_IMAGE\"}"

