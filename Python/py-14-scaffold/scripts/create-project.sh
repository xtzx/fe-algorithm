#!/bin/bash
# 创建新项目脚本
#
# 用法: ./create-project.sh my-project
#
# 此脚本将复制模板并替换项目名称

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$(dirname "$SCRIPT_DIR")"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <project-name>"
    echo "示例: $0 my-awesome-project"
    exit 1
fi

PROJECT_NAME="$1"
PROJECT_DIR="$(pwd)/$PROJECT_NAME"

# 检查目录是否已存在
if [ -d "$PROJECT_DIR" ]; then
    echo "错误: 目录已存在: $PROJECT_DIR"
    exit 1
fi

echo "=================================================="
echo "创建新项目: $PROJECT_NAME"
echo "=================================================="

# 创建项目目录
mkdir -p "$PROJECT_DIR"

# 复制模板文件
echo "复制模板文件..."
cp -r "$TEMPLATE_DIR/src" "$PROJECT_DIR/"
cp -r "$TEMPLATE_DIR/tests" "$PROJECT_DIR/"
cp -r "$TEMPLATE_DIR/scripts" "$PROJECT_DIR/"
cp "$TEMPLATE_DIR/pyproject.toml" "$PROJECT_DIR/"
cp "$TEMPLATE_DIR/.python-version" "$PROJECT_DIR/" 2>/dev/null || true
cp "$TEMPLATE_DIR/.pre-commit-config.yaml" "$PROJECT_DIR/"
cp "$TEMPLATE_DIR/env.example" "$PROJECT_DIR/.env.example" 2>/dev/null || true

# 重命名包目录
if [ -d "$PROJECT_DIR/src/scaffold" ]; then
    # 将 scaffold 替换为项目名（转换为 snake_case）
    PACKAGE_NAME=$(echo "$PROJECT_NAME" | tr '-' '_')
    mv "$PROJECT_DIR/src/scaffold" "$PROJECT_DIR/src/$PACKAGE_NAME"

    # 替换代码中的包名
    if command -v sed &> /dev/null; then
        find "$PROJECT_DIR" -type f -name "*.py" -exec sed -i.bak "s/scaffold/$PACKAGE_NAME/g" {} \;
        find "$PROJECT_DIR" -type f -name "*.py.bak" -delete
        find "$PROJECT_DIR" -type f -name "*.toml" -exec sed -i.bak "s/scaffold/$PACKAGE_NAME/g" {} \;
        find "$PROJECT_DIR" -type f -name "*.toml.bak" -delete
    fi
fi

# 创建 README.md
cat > "$PROJECT_DIR/README.md" << EOF
# $PROJECT_NAME

## 快速开始

\`\`\`bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 复制环境配置
cp .env.example .env

# 运行测试
pytest

# 运行应用
python -m $PACKAGE_NAME --help
\`\`\`

## 开发

\`\`\`bash
# 格式化代码
./scripts/format.sh

# 代码检查
./scripts/lint.sh

# 类型检查
./scripts/typecheck.sh

# 运行测试
./scripts/test.sh
\`\`\`
EOF

# 初始化 git（可选）
if command -v git &> /dev/null; then
    cd "$PROJECT_DIR"
    git init -q
    echo ".venv/" > .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".env" >> .gitignore
    echo "*.egg-info/" >> .gitignore
    echo "dist/" >> .gitignore
    echo "build/" >> .gitignore
    echo "htmlcov/" >> .gitignore
    echo ".coverage" >> .gitignore
    echo ".pytest_cache/" >> .gitignore
    echo ".ruff_cache/" >> .gitignore
fi

echo ""
echo "=================================================="
echo "项目创建完成: $PROJECT_DIR"
echo "=================================================="
echo ""
echo "下一步:"
echo "  cd $PROJECT_NAME"
echo "  python -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -e \".[dev]\""
echo "  pre-commit install"
echo ""

