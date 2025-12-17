#!/bin/bash

# ============================================
# Webpack 构建脚本（使用自定义插件）
#
# 用途：演示如何使用自定义 Webpack 插件
#
# 使用方法：
#   chmod +x run-webpack-with-plugin.sh
#   ./run-webpack-with-plugin.sh
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
WEBPACK_PLUGINS_DIR="$PROJECT_ROOT/webpack-plugins"
BUILD_DIR="$PROJECT_ROOT/dist"

# ============================================
# 步骤 1: 准备环境
# ============================================

log_step "1/5 准备构建环境..."

cd "$PROJECT_ROOT"

# 检查 package.json
if [ ! -f "package.json" ]; then
    log_info "初始化 package.json..."
    npm init -y > /dev/null 2>&1
fi

# 安装 Webpack 依赖
if [ ! -d "node_modules/webpack" ]; then
    log_info "安装 Webpack 依赖..."
    npm install --save-dev webpack webpack-cli > /dev/null 2>&1
fi

log_info "环境准备完成 ✓"

# ============================================
# 步骤 2: 创建测试入口文件
# ============================================

log_step "2/5 创建测试入口文件..."

mkdir -p src

cat > src/index.js << 'EOF'
// 测试入口文件
console.log('Hello from Webpack!');

// 模拟一些代码
const add = (a, b) => a + b;
const subtract = (a, b) => a - b;

export { add, subtract };
EOF

log_info "入口文件创建完成 ✓"

# ============================================
# 步骤 3: 创建 Webpack 配置
# ============================================

log_step "3/5 创建 Webpack 配置..."

cat > webpack.config.js << EOF
const path = require('path');
const SimpleBuildInfoPlugin = require('./webpack-plugins/simple-build-info-plugin');
const BundleSizeReportPlugin = require('./webpack-plugins/bundle-size-report-plugin');

module.exports = {
  mode: 'production',

  entry: './src/index.js',

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.[contenthash:8].js',
    clean: true
  },

  plugins: [
    // 构建信息插件
    new SimpleBuildInfoPlugin({
      outputFile: 'build-info.json'
    }),

    // 体积报告插件
    new BundleSizeReportPlugin({
      threshold: 10 * 1024,  // 10KB（为了演示，设置较小的阈值）
      showDetails: true
    })
  ]
};
EOF

log_info "Webpack 配置创建完成 ✓"

# ============================================
# 步骤 4: 执行构建
# ============================================

log_step "4/5 执行 Webpack 构建..."

echo ""
npx webpack --config webpack.config.js

# ============================================
# 步骤 5: 查看输出
# ============================================

log_step "5/5 查看构建产物..."

echo ""
log_info "构建产物列表:"
ls -la "$BUILD_DIR"

echo ""
log_info "build-info.json 内容:"
cat "$BUILD_DIR/build-info.json"

# ============================================
# 完成
# ============================================

echo ""
log_info "=========================================="
log_info "构建完成！"
log_info "=========================================="
log_info "输出目录: $BUILD_DIR"
log_info ""
log_info "插件效果："
log_info "1. SimpleBuildInfoPlugin: 生成了 build-info.json"
log_info "2. BundleSizeReportPlugin: 在控制台输出了体积报告"
log_info "=========================================="


# ============================================
# 清理（可选）
# ============================================

# 如果需要清理测试文件：
# rm -rf src dist webpack.config.js

