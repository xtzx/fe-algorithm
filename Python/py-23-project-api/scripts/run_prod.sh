#!/bin/bash

# 生产模式启动

set -e

cd "$(dirname "$0")/.."

echo "=== 启动生产服务器 ==="

# 设置环境变量
export APP_ENV=production
export DEBUG=false
export LOG_FORMAT=json

# 使用 gunicorn + uvicorn workers
gunicorn bookmark_api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile -

