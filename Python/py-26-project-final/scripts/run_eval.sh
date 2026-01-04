#!/bin/bash
# ============================================
# 评测运行脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}运行评测...${NC}"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 默认参数
API_URL="${API_URL:-http://localhost:8000}"
DATASET="${DATASET:-./data/eval_dataset/sample.json}"
OUTPUT="${OUTPUT:-./data/eval_results.json}"

# 创建示例数据集（如果不存在）
if [ ! -f "$DATASET" ]; then
    echo -e "${YELLOW}创建示例数据集...${NC}"
    python -m knowledge_assistant.evaluation.cli create-dataset -o "$DATASET"
fi

# 运行评测
echo -e "${GREEN}开始评测 (API: $API_URL)${NC}"
python -m knowledge_assistant.evaluation.cli run \
    -d "$DATASET" \
    -o "$OUTPUT" \
    --api-url "$API_URL"

echo -e "${GREEN}评测完成！${NC}"
echo -e "结果保存在: $OUTPUT"


