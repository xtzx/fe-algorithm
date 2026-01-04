#!/bin/bash
# 运行演示脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$PROJECT_DIR/examples"

cd "$PROJECT_DIR"

echo "=================================================="
echo "文件自动化工具演示"
echo "=================================================="
echo

# 创建示例文件目录
DEMO_DIR="$EXAMPLES_DIR/demo_files"
if [ -d "$DEMO_DIR" ]; then
    rm -rf "$DEMO_DIR"
fi
mkdir -p "$DEMO_DIR"

# 创建示例文件
echo "创建示例文件..."
for i in $(seq 1 5); do
    echo "Report content $i" > "$DEMO_DIR/report_2024010$i.txt"
done
echo "Image data" > "$DEMO_DIR/photo.jpg"
echo "Document content" > "$DEMO_DIR/readme.pdf"
echo '{"data": "json"}' > "$DEMO_DIR/data.json"
echo "Code here" > "$DEMO_DIR/script.py"
echo "Temp file" > "$DEMO_DIR/temp.tmp"

echo "创建完成！目录内容："
ls -la "$DEMO_DIR"
echo

# 演示 1: 重命名预览
echo "=================================================="
echo "演示 1: 批量重命名（预览模式）"
echo "=================================================="
echo "命令: python -m file_automation rename $DEMO_DIR --mode regex --pattern 'report_(\\d+)' --replacement '\\1_report' --glob '*.txt'"
echo
python -m file_automation rename "$DEMO_DIR" \
    --mode regex \
    --pattern 'report_(\d+)' \
    --replacement '\1_report' \
    --glob '*.txt'
echo

# 演示 2: 文件分类预览
echo "=================================================="
echo "演示 2: 文件分类（预览模式）"
echo "=================================================="
echo "命令: python -m file_automation organize $DEMO_DIR --by extension"
echo
python -m file_automation organize "$DEMO_DIR" --by extension
echo

# 演示 3: 清理预览
echo "=================================================="
echo "演示 3: 清理临时文件（预览模式）"
echo "=================================================="
echo "命令: python -m file_automation clean $DEMO_DIR --patterns '*.tmp'"
echo
python -m file_automation clean "$DEMO_DIR" --patterns '*.tmp'
echo

echo "=================================================="
echo "演示完成！"
echo "=================================================="
echo
echo "以上都是预览模式，添加 --execute 或 -x 实际执行"
echo "例如: python -m file_automation organize $DEMO_DIR --by extension --execute"

