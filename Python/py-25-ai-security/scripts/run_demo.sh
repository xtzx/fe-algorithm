#!/bin/bash

# 运行演示

set -e

cd "$(dirname "$0")/.."

echo "=== AI 安全演示 ==="
echo ""

echo "1. 安全防护演示"
echo "---------------"
python examples/security_demo.py

echo ""
echo "2. 评测系统演示"
echo "---------------"
python examples/evaluation_demo.py


