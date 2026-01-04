#!/usr/bin/env python3
"""
02_lambda.py - lambda 表达式演示
"""

# =============================================================================
# 1. 基本语法
# =============================================================================
print("=== lambda 基本语法 ===")

# 单参数
square = lambda x: x**2
print(f"square(5) = {square(5)}")

# 多参数
add = lambda x, y: x + y
print(f"add(3, 5) = {add(3, 5)}")

# 无参数
get_answer = lambda: 42
print(f"get_answer() = {get_answer()}")

# =============================================================================
# 2. 作为排序的 key
# =============================================================================
print("\n=== lambda 作为 key ===")

words = ["apple", "pie", "banana"]
sorted_by_len = sorted(words, key=lambda w: len(w))
print(f"按长度排序: {sorted_by_len}")

pairs = [(1, 3), (2, 1), (3, 2)]
sorted_by_second = sorted(pairs, key=lambda p: p[1])
print(f"按第二个元素排序: {sorted_by_second}")

# =============================================================================
# 3. 与 map/filter 配合
# =============================================================================
print("\n=== lambda 与 map/filter ===")

numbers = [1, 2, 3, 4, 5]

squares = list(map(lambda x: x**2, numbers))
print(f"map(lambda x: x**2, {numbers}) = {squares}")

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter(lambda x: x%2==0, {numbers}) = {evens}")

# =============================================================================
# 4. 条件表达式
# =============================================================================
print("\n=== lambda 条件表达式 ===")

max_val = lambda a, b: a if a > b else b
print(f"max_val(5, 3) = {max_val(5, 3)}")

is_even = lambda x: x % 2 == 0
print(f"is_even(4) = {is_even(4)}")

# =============================================================================
# 5. 嵌套 lambda
# =============================================================================
print("\n=== 嵌套 lambda ===")

make_adder = lambda n: lambda x: x + n
add_5 = make_adder(5)
print(f"add_5(10) = {add_5(10)}")

# =============================================================================
# 6. 常见陷阱：循环中的 lambda
# =============================================================================
print("\n=== 循环中的 lambda 陷阱 ===")

# ❌ 问题
funcs = []
for i in range(3):
    funcs.append(lambda x: x + i)

print("❌ 错误方式（所有捕获最后一个值）:")
for f in funcs:
    print(f"  f(10) = {f(10)}")

# ✅ 解决：使用默认参数
funcs = []
for i in range(3):
    funcs.append(lambda x, i=i: x + i)

print("\n✅ 正确方式（使用默认参数）:")
for f in funcs:
    print(f"  f(10) = {f(10)}")

print("\n=== 运行完成 ===")

