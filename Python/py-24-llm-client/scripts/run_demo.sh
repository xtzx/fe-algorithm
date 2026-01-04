#!/bin/bash

# 运行演示

set -e

cd "$(dirname "$0")/.."

echo "=== LLM Kit 演示 ==="
echo ""

echo "1. RAG 演示"
echo "-----------"
python examples/rag_demo.py

echo ""
echo "2. 流式输出演示"
echo "---------------"
python examples/streaming_demo.py


