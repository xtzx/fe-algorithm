#!/bin/bash
# 运行所有示例

echo "🚀 运行 py-04-functional 所有示例"
echo "=================================="

cd "$(dirname "$0")/.."

echo ""
echo "1. 高阶函数示例"
python3 examples/01_higher_order_functions.py

echo ""
echo "2. lambda 表达式示例"
python3 examples/02_lambda.py

echo ""
echo "3. 闭包示例"
python3 examples/03_closure.py

echo ""
echo "4. 装饰器示例"
python3 examples/04_decorators.py

echo ""
echo "5. 生成器示例"
python3 examples/05_generators.py

echo ""
echo "6. functools 和 itertools 示例"
python3 examples/06_functools_itertools.py

echo ""
echo "=================================="
echo "✅ 所有示例运行完成"

