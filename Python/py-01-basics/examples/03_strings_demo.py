#!/usr/bin/env python3
"""
03_strings_demo.py - 字符串操作演示

演示：
- 字符串创建
- f-string 格式化
- 索引与切片
- 常用方法
"""

# =============================================================================
# 1. 字符串创建
# =============================================================================

print("=== 字符串创建 ===")

single = 'Hello'
double = "World"
multi = """
这是
多行
字符串
"""
raw = r"C:\Users\name"  # 原始字符串，不转义

print(f"单引号: {single}")
print(f"双引号: {double}")
print(f"多行: {multi}")
print(f"原始字符串: {raw}")

# =============================================================================
# 2. f-string 格式化（推荐）
# =============================================================================

print("=== f-string 格式化 ===")

name = "Alice"
age = 25
pi = 3.14159

# 基本用法
print(f"Hello, {name}!")
print(f"{name} is {age} years old")

# 表达式
print(f"2 + 2 = {2 + 2}")
print(f"大写: {name.upper()}")

# 格式化数字
print(f"Pi: {pi:.2f}")           # 2 位小数
print(f"百分比: {0.756:.1%}")    # 百分比
print(f"填充: {42:05d}")         # 前导零
print(f"千分位: {1234567:,}")    # 千分位

# 对齐
print(f"|{name:<10}|")  # 左对齐
print(f"|{name:>10}|")  # 右对齐
print(f"|{name:^10}|")  # 居中

# =============================================================================
# 3. 索引与切片
# =============================================================================

print("\n=== 索引与切片 ===")

s = "Hello World"
print(f"字符串: '{s}'")

# 索引
print(f"s[0] = '{s[0]}'")    # 第一个
print(f"s[-1] = '{s[-1]}'")  # 最后一个

# 切片 [start:end]
print(f"s[0:5] = '{s[0:5]}'")   # Hello
print(f"s[6:] = '{s[6:]}'")    # World
print(f"s[:5] = '{s[:5]}'")    # Hello

# 步长 [start:end:step]
print(f"s[::2] = '{s[::2]}'")    # 每隔一个
print(f"s[::-1] = '{s[::-1]}'")  # 反转

# =============================================================================
# 4. 常用方法
# =============================================================================

print("\n=== 常用方法 ===")

text = "  Hello World  "

# 大小写
print(f"upper(): '{text.upper()}'")
print(f"lower(): '{text.lower()}'")
print(f"title(): '{text.title()}'")
print(f"capitalize(): '{text.capitalize()}'")

# 去空白
print(f"strip(): '{text.strip()}'")
print(f"lstrip(): '{text.lstrip()}'")
print(f"rstrip(): '{text.rstrip()}'")

# 查找替换
s = "Hello World"
print(f"\nfind('World'): {s.find('World')}")
print(f"find('Python'): {s.find('Python')}")
print(f"replace('World', 'Python'): '{s.replace('World', 'Python')}'")
print(f"count('l'): {s.count('l')}")

# 分割连接
csv = "a,b,c,d"
items = csv.split(",")
print(f"\nsplit(','): {items}")
print(f"'-'.join(items): '{'-'.join(items)}'")

# 判断方法
print(f"\n'Hello'.startswith('He'): {'Hello'.startswith('He')}")
print(f"'Hello'.endswith('lo'): {'Hello'.endswith('lo')}")
print(f"'123'.isdigit(): {'123'.isdigit()}")
print(f"'abc'.isalpha(): {'abc'.isalpha()}")

# =============================================================================
# 5. 字符串不可变
# =============================================================================

print("\n=== 字符串不可变 ===")

s = "hello"
print(f"原字符串: '{s}'")

# s[0] = "H"  # ❌ 会报错

# 只能创建新字符串
s = "H" + s[1:]
print(f"修改后: '{s}'")

# =============================================================================
# 6. 编码
# =============================================================================

print("\n=== 编码 ===")

chinese = "你好"
encoded = chinese.encode("utf-8")
decoded = encoded.decode("utf-8")

print(f"原字符串: {chinese}")
print(f"UTF-8 编码: {encoded}")
print(f"解码: {decoded}")

print("\n=== 运行完成 ===")

