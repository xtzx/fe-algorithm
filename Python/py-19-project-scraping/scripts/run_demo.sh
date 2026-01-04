#!/bin/bash
# 运行演示

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "  技术博客聚合器 - 演示"
echo "=================================================="
echo

# 确保数据目录存在
mkdir -p data

echo "1. 采集 Hacker News 文章..."
echo "--------------------------------------------------"
python -m blog_aggregator collect --source hackernews --max-pages 1

echo
echo "2. 查看状态..."
echo "--------------------------------------------------"
python -m blog_aggregator status

echo
echo "3. 生成终端报告..."
echo "--------------------------------------------------"
python -m blog_aggregator report --format terminal --limit 10

echo
echo "4. 生成 Markdown 报告..."
echo "--------------------------------------------------"
python -m blog_aggregator report --format markdown --output data/report.md
echo "报告已保存到 data/report.md"

echo
echo "=================================================="
echo "  演示完成!"
echo "=================================================="
echo
echo "更多命令:"
echo "  python -m blog_aggregator collect --all         # 采集所有源"
echo "  python -m blog_aggregator collect -i            # 增量采集"
echo "  python -m blog_aggregator report -f json        # JSON 报告"
echo "  python -m blog_aggregator status                # 查看状态"

