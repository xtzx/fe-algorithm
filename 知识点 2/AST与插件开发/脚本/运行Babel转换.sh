#!/bin/bash

# ============================================
# Babel 转换脚本
#
# 用途：演示如何使用自定义 Babel 插件转换代码
#
# 使用方法：
#   chmod +x run-babel-transform.sh
#   ./run-babel-transform.sh
# ============================================

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# ============================================
# 配置
# ============================================

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BABEL_PLUGINS_DIR="$PROJECT_ROOT/babel-plugins"
EXAMPLES_DIR="$BABEL_PLUGINS_DIR/examples"
INPUT_FILE="$EXAMPLES_DIR/input-sample.js"
OUTPUT_FILE="$EXAMPLES_DIR/output-transformed.js"

# ============================================
# 步骤 1: 检查依赖
# ============================================

log_step "1/4 检查 Babel 依赖..."

if ! command -v npx &> /dev/null; then
    echo "Error: npx 未找到，请安装 Node.js"
    exit 1
fi

# 检查是否有 package.json，如果没有就初始化
if [ ! -f "$PROJECT_ROOT/package.json" ]; then
    log_info "初始化 package.json..."
    cd "$PROJECT_ROOT"
    npm init -y > /dev/null 2>&1
fi

# 安装 Babel 依赖（如果需要）
if [ ! -d "$PROJECT_ROOT/node_modules/@babel" ]; then
    log_info "安装 Babel 依赖..."
    cd "$PROJECT_ROOT"
    npm install --save-dev @babel/core @babel/cli @babel/preset-env > /dev/null 2>&1
fi

log_info "依赖检查完成 ✓"

# ============================================
# 步骤 2: 显示输入文件
# ============================================

log_step "2/4 输入文件内容 (前 30 行):"
echo ""
head -30 "$INPUT_FILE"
echo ""
echo "..."
echo ""

# ============================================
# 步骤 3: 执行 Babel 转换
# ============================================

log_step "3/4 执行 Babel 转换..."

# 使用 log-inject-plugin 插件
cd "$PROJECT_ROOT"

npx babel "$INPUT_FILE" \
    --plugins "$BABEL_PLUGINS_DIR/log-inject-plugin.js" \
    --out-file "$OUTPUT_FILE"

log_info "转换完成，输出到: $OUTPUT_FILE"

# ============================================
# 步骤 4: 显示输出文件
# ============================================

log_step "4/4 输出文件内容 (前 30 行):"
echo ""
head -30 "$OUTPUT_FILE"
echo ""
echo "..."
echo ""

# ============================================
# 对比结果
# ============================================

echo ""
log_info "=========================================="
log_info "转换完成！"
log_info "=========================================="
log_info "输入文件: $INPUT_FILE"
log_info "输出文件: $OUTPUT_FILE"
log_info ""
log_info "变化说明："
log_info "- 所有 track() 调用都添加了 __source 参数"
log_info "- __source 值为当前文件名"
log_info "=========================================="


# ============================================
# 可选：使用 Node API 执行转换
# ============================================

# 如果需要更多控制，可以使用 Node API：
#
# node -e "
# const babel = require('@babel/core');
# const fs = require('fs');
#
# const code = fs.readFileSync('$INPUT_FILE', 'utf-8');
#
# const result = babel.transformSync(code, {
#   plugins: ['$BABEL_PLUGINS_DIR/log-inject-plugin.js'],
#   filename: 'input-sample.js'
# });
#
# console.log(result.code);
# "

