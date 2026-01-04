#!/bin/bash
# 运行所有示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "  asyncio 并发 - 示例"
echo "=================================================="
echo

echo "1. 基础示例"
echo "--------------------------------------------------"
python examples/demo_basics.py

echo
echo "2. 实战模式示例"
echo "--------------------------------------------------"
python examples/demo_patterns.py

echo
echo "3. 生产者/消费者示例"
echo "--------------------------------------------------"
python examples/demo_producer_consumer.py

echo
echo "=================================================="
echo "  示例完成!"
echo "=================================================="

