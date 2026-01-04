#!/bin/bash
# 日志分析器演示脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "  日志分析与清理工具 - 演示"
echo "=================================================="
echo

# 1. 分析 Nginx 日志
echo "📊 分析 Nginx 访问日志..."
echo "----------------------------------------"
python -m log_analyzer analyze sample_logs/nginx.log --format nginx --verbose
echo

# 2. 分析应用日志
echo "📊 分析应用日志..."
echo "----------------------------------------"
python -m log_analyzer analyze sample_logs/app.log --format app
echo

# 3. 分析 JSON 日志
echo "📊 分析 JSON 日志..."
echo "----------------------------------------"
python -m log_analyzer analyze sample_logs/json.log --format json
echo

# 4. 生成 Markdown 报告
echo "📋 生成 Markdown 报告..."
echo "----------------------------------------"
python -m log_analyzer report sample_logs/ --format markdown --output report.md
echo "报告已保存到 report.md"
echo

# 5. 生成 JSON 报告
echo "📋 生成 JSON 报告..."
echo "----------------------------------------"
python -m log_analyzer report sample_logs/ --format json --output report.json
echo "报告已保存到 report.json"
echo

# 6. 清理预览（dry-run）
echo "🧹 清理预览 (dry-run)..."
echo "----------------------------------------"
python -m log_analyzer clean sample_logs/ --older-than 1 --dry-run || echo "(No files older than 1 day)"
echo

echo "=================================================="
echo "  演示完成!"
echo "=================================================="

