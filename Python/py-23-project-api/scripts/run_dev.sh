#!/bin/bash

# 开发模式启动

set -e

cd "$(dirname "$0")/.."

echo "=== 启动开发服务器 ==="

# 设置环境变量
export DEBUG=true
export LOG_FORMAT=console

# 启动服务器
uvicorn bookmark_api.main:app --reload --host 0.0.0.0 --port 8000

