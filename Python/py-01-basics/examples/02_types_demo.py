#!/usr/bin/env python3
"""
02_types_demo.py - 变量与类型演示

演示：
- 变量声明
- 基本数据类型
- 类型转换
- 类型检查
"""

# =============================================================================
# 1. 变量声明（无需关键字）
# =============================================================================

name = "Alice"      # 字符串
age = 25            # 整数
height = 1.68       # 浮点数
is_student = True   # 布尔值
data = None         # 空值

print(f"姓名: {name}")
print(f"年龄: {age}")
print(f"身高: {height}")
print(f"是学生: {is_student}")
print(f"数据: {data}")

# =============================================================================
# 2. 多重赋值
# =============================================================================

x, y, z = 1, 2, 3
print(f"\nx={x}, y={y}, z={z}")

# 交换变量（Python 特色）
a, b = 10, 20
print(f"交换前: a={a}, b={b}")
a, b = b, a
print(f"交换后: a={a}, b={b}")

# =============================================================================
# 3. 数据类型
# =============================================================================

print("\n=== 数据类型 ===")

# 整数（无大小限制）
big_num = 10 ** 100
print(f"大整数: {big_num}")

# 不同进制
binary = 0b1010      # 二进制
octal = 0o17         # 八进制
hexadecimal = 0xFF   # 十六进制
print(f"二进制 0b1010 = {binary}")
print(f"八进制 0o17 = {octal}")
print(f"十六进制 0xFF = {hexadecimal}")

# 可读性分隔符（Python 3.6+）
million = 1_000_000
print(f"一百万: {million}")

# 浮点数
pi = 3.14159
scientific = 2.5e-3  # 0.0025
print(f"π ≈ {pi}")
print(f"科学计数法 2.5e-3 = {scientific}")

# 布尔值（注意大小写）
print(f"\nTrue 的类型: {type(True)}")
print(f"False 的类型: {type(False)}")

# =============================================================================
# 4. 类型转换
# =============================================================================

print("\n=== 类型转换 ===")

# 转整数
print(f'int("42") = {int("42")}')
print(f"int(3.7) = {int(3.7)}")  # 截断
print(f'int("10", 2) = {int("10", 2)}')  # 二进制

# 转浮点
print(f'float("3.14") = {float("3.14")}')
print(f"float(42) = {float(42)}")

# 转字符串
print(f"str(42) = '{str(42)}'")
print(f"str(True) = '{str(True)}'")

# 转布尔
print(f"\nbool(0) = {bool(0)}")
print(f"bool('') = {bool('')}")
print(f"bool([]) = {bool([])}")
print(f"bool(1) = {bool(1)}")
print(f"bool('hello') = {bool('hello')}")

# =============================================================================
# 5. 类型检查
# =============================================================================

print("\n=== 类型检查 ===")

# type()
print(f"type(42) = {type(42)}")
print(f"type(3.14) = {type(3.14)}")
print(f"type('hello') = {type('hello')}")
print(f"type(True) = {type(True)}")
print(f"type(None) = {type(None)}")

# isinstance()
print(f"\nisinstance(42, int) = {isinstance(42, int)}")
print(f"isinstance(42, (int, float)) = {isinstance(42, (int, float))}")
print(f"isinstance(True, int) = {isinstance(True, int)}")  # bool 是 int 子类！

# =============================================================================
# 6. Truthy 和 Falsy（与 JS 的区别）
# =============================================================================

print("\n=== Falsy 值 ===")

falsy_values = [False, None, 0, 0.0, "", [], {}, set()]

for val in falsy_values:
    print(f"bool({val!r:10}) = {bool(val)}")

print("\n⚠️ 注意: Python 空列表/字典是 Falsy，JS 是 Truthy！")

print("\n=== 运行完成 ===")

