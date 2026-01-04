#!/bin/bash
# 运行所有示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "  HTTP 客户端工程化 - 示例"
echo "=================================================="
echo

# 运行高级功能示例（不需要网络）
echo "运行 http_kit 高级功能示例..."
echo "--------------------------------------------------"
python examples/advanced_features.py

echo
echo "=================================================="
echo "  示例完成!"
echo "=================================================="
echo
echo "其他示例（需要网络）:"
echo "  python examples/basic_usage.py"
echo "  python examples/async_usage.py"

