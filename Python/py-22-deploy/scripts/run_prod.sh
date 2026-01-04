#!/bin/bash

# ============================================
# 生产环境运行脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 配置
APP_MODULE="${APP_MODULE:-main:app}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo -e "${BLUE}=== 生产环境启动 ===${NC}"

# 检查是否安装了 gunicorn
if ! command -v gunicorn &> /dev/null; then
    echo -e "${YELLOW}警告: gunicorn 未安装，使用 uvicorn${NC}"
    
    uvicorn "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
else
    echo "启动参数:"
    echo "  应用: $APP_MODULE"
    echo "  地址: $HOST:$PORT"
    echo "  Workers: $WORKERS"
    echo "  日志级别: $LOG_LEVEL"
    echo ""
    
    gunicorn "$APP_MODULE" \
        --workers "$WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind "$HOST:$PORT" \
        --timeout 120 \
        --graceful-timeout 30 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --access-logfile - \
        --error-logfile - \
        --log-level "$LOG_LEVEL" \
        --capture-output
fi

