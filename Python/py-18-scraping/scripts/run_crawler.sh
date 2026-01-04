#!/bin/bash
# 运行爬虫示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "  爬虫工程化 - 示例"
echo "=================================================="
echo

# 运行 demo（不需要网络）
echo "1. 运行解析演示..."
echo "--------------------------------------------------"
python examples/simple_crawler.py

echo
echo "=================================================="
echo "  示例完成!"
echo "=================================================="
echo
echo "爬取真实网站:"
echo "  python examples/simple_crawler.py https://example.com --max-pages 10"
echo
echo "使用 CLI:"
echo "  python -m scraper crawl https://example.com --max-pages 10"

