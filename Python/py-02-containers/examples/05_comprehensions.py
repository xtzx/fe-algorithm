#!/usr/bin/env python3
"""
05_comprehensions.py - 推导式演示
"""

# =============================================================================
# 1. 列表推导式
# =============================================================================
print("=== 列表推导式 ===")

# 基本
squares = [x**2 for x in range(10)]
print(f"平方: {squares}")

# 条件过滤
evens = [x for x in range(20) if x % 2 == 0]
print(f"偶数: {evens}")

# 条件表达式
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(f"标签: {labels}")

# 嵌套循环
pairs = [(x, y) for x in range(3) for y in range(3)]
print(f"组合: {pairs}")

# 展平嵌套列表
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"展平: {flat}")

# =============================================================================
# 2. 字典推导式
# =============================================================================
print("\n=== 字典推导式 ===")

# 平方映射
square_dict = {x: x**2 for x in range(5)}
print(f"平方字典: {square_dict}")

# 从两个列表
keys = ["a", "b", "c"]
values = [1, 2, 3]
d = {k: v for k, v in zip(keys, values)}
print(f"zip 创建: {d}")

# 条件过滤
filtered = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"过滤: {filtered}")

# 键值互换
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(f"互换: {inverted}")

# 单词长度
words = ["hello", "world", "python"]
lengths = {word: len(word) for word in words}
print(f"长度: {lengths}")

# =============================================================================
# 3. 集合推导式
# =============================================================================
print("\n=== 集合推导式 ===")

# 平方（自动去重）
squares = {x**2 for x in range(-5, 6)}
print(f"平方集合: {squares}")

# 字符集合
chars = {c.lower() for c in "Hello World"}
print(f"字符集合: {chars}")

# =============================================================================
# 4. 生成器表达式
# =============================================================================
print("\n=== 生成器表达式 ===")

# 创建生成器
gen = (x**2 for x in range(10))
print(f"生成器对象: {gen}")

# 转为列表
lst = list(x**2 for x in range(5))
print(f"转为列表: {lst}")

# 用于函数参数
total = sum(x**2 for x in range(10))
print(f"平方和: {total}")

max_len = max(len(w) for w in ["hello", "world", "python"])
print(f"最大长度: {max_len}")

# 判断
has_even = any(x % 2 == 0 for x in [1, 2, 3])
all_positive = all(x > 0 for x in [1, 2, 3])
print(f"有偶数: {has_even}, 全正数: {all_positive}")

# =============================================================================
# 5. 嵌套推导式
# =============================================================================
print("\n=== 嵌套推导式 ===")

# 生成矩阵
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print("矩阵:")
for row in matrix:
    print(f"  {row}")

# 矩阵转置
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print("转置:")
for row in transposed:
    print(f"  {row}")

# =============================================================================
# 6. 推导式 vs map/filter
# =============================================================================
print("\n=== 推导式 vs map/filter ===")

numbers = [1, 2, 3, 4, 5]

# map
map_result = list(map(lambda x: x**2, numbers))
comp_result = [x**2 for x in numbers]
print(f"map: {map_result}")
print(f"推导式: {comp_result}")

# filter
filter_result = list(filter(lambda x: x % 2 == 0, numbers))
comp_result = [x for x in numbers if x % 2 == 0]
print(f"filter: {filter_result}")
print(f"推导式: {comp_result}")

print("\n=== 运行完成 ===")

