#!/usr/bin/env python3
"""
06_file_io.py - 文件 I/O 演示

演示：
- print() 高级用法
- 文件读写
- with 语句
- pathlib
"""

import json
from pathlib import Path

# =============================================================================
# 1. print() 高级用法
# =============================================================================

print("=== print() 高级用法 ===")

# sep 参数
print("a", "b", "c", sep=" | ")

# end 参数
print("Loading", end="")
print(".", end="")
print(".", end="")
print(". Done!")

# 格式化输出
name = "Alice"
age = 25
print(f"|{name:<10}|{age:>5}|")
print(f"|{name:^10}|{age:^5}|")

# =============================================================================
# 2. 创建测试文件
# =============================================================================

print("\n=== 文件写入 ===")

# 确保目录存在
test_dir = Path(__file__).parent / "test_files"
test_dir.mkdir(exist_ok=True)

# 写入文本文件
test_file = test_dir / "sample.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("Hello, Python!\n")
    f.write("这是第二行\n")
    f.write("This is the third line\n")

print(f"文件已创建: {test_file}")

# =============================================================================
# 3. 文件读取
# =============================================================================

print("\n=== 文件读取 ===")

# read() - 读取全部
with open(test_file, "r", encoding="utf-8") as f:
    content = f.read()
    print("read() 结果:")
    print(content)

# readlines() - 读取所有行
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print("readlines() 结果:")
    print(lines)

# 逐行迭代（推荐大文件）
print("逐行迭代:")
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        print(f"  {line.strip()}")

# =============================================================================
# 4. 追加模式
# =============================================================================

print("\n=== 追加模式 ===")

with open(test_file, "a", encoding="utf-8") as f:
    f.write("追加的新行\n")

with open(test_file, "r", encoding="utf-8") as f:
    print(f.read())

# =============================================================================
# 5. pathlib 操作
# =============================================================================

print("=== pathlib ===")

path = Path(test_file)

print(f"文件存在: {path.exists()}")
print(f"是文件: {path.is_file()}")
print(f"文件名: {path.name}")
print(f"扩展名: {path.suffix}")
print(f"父目录: {path.parent}")

# 读写（pathlib 方式）
content = path.read_text(encoding="utf-8")
print(f"行数: {len(content.splitlines())}")

# =============================================================================
# 6. JSON 文件
# =============================================================================

print("\n=== JSON 文件 ===")

json_file = test_dir / "data.json"

# 写入 JSON
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding"],
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"JSON 文件已创建: {json_file}")

# 读取 JSON
with open(json_file, "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

print(f"读取的数据: {loaded_data}")
print(f"姓名: {loaded_data['name']}")

# =============================================================================
# 7. 目录操作
# =============================================================================

print("\n=== 目录操作 ===")

# 列出目录内容
print(f"{test_dir} 中的文件:")
for item in test_dir.iterdir():
    print(f"  {item.name}")

# 查找特定文件
print("\n所有 .txt 文件:")
for txt_file in test_dir.glob("*.txt"):
    print(f"  {txt_file.name}")

# =============================================================================
# 8. 清理测试文件
# =============================================================================

print("\n=== 清理 ===")

# 删除测试文件
for file in test_dir.iterdir():
    file.unlink()
    print(f"删除: {file.name}")

test_dir.rmdir()
print(f"删除目录: {test_dir}")

print("\n=== 运行完成 ===")

