#!/usr/bin/env python3
"""
01_higher_order_functions.py - 高阶函数演示
"""

from functools import reduce

# =============================================================================
# 1. 函数作为参数
# =============================================================================
print("=== 函数作为参数 ===")

def apply(func, value):
    """应用函数到值"""
    return func(value)

def square(x):
    return x**2

result = apply(square, 5)
print(f"apply(square, 5) = {result}")

# lambda 函数
result = apply(lambda x: x**2, 5)
print(f"apply(lambda x: x**2, 5) = {result}")

# =============================================================================
# 2. 函数作为返回值
# =============================================================================
print("\n=== 函数作为返回值 ===")

def make_multiplier(n):
    """创建乘法器"""
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")

# =============================================================================
# 3. map
# =============================================================================
print("\n=== map ===")

numbers = [1, 2, 3, 4, 5]

# 使用 map
squares = list(map(lambda x: x**2, numbers))
print(f"map(lambda x: x**2, {numbers}) = {squares}")

# 多个可迭代对象
a = [1, 2, 3]
b = [4, 5, 6]
sums = list(map(lambda x, y: x + y, a, b))
print(f"map(lambda x, y: x+y, {a}, {b}) = {sums}")

# =============================================================================
# 4. filter
# =============================================================================
print("\n=== filter ===")

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 过滤偶数
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter(lambda x: x%2==0, {numbers}) = {evens}")

# 过滤 Falsy 值
data = [0, 1, "", "hello", None, [], [1, 2]]
clean = list(filter(None, data))
print(f"filter(None, {data}) = {clean}")

# =============================================================================
# 5. reduce
# =============================================================================
print("\n=== reduce ===")

numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda acc, x: acc + x, numbers)
print(f"reduce(lambda acc, x: acc+x, {numbers}) = {total}")

# 求积
product = reduce(lambda acc, x: acc * x, numbers)
print(f"reduce(lambda acc, x: acc*x, {numbers}) = {product}")

# 带初始值
total = reduce(lambda acc, x: acc + x, numbers, 10)
print(f"reduce(lambda acc, x: acc+x, {numbers}, 10) = {total}")

# =============================================================================
# 6. sorted 的 key 参数
# =============================================================================
print("\n=== sorted with key ===")

words = ["apple", "pie", "banana", "cherry"]
sorted_by_len = sorted(words, key=len)
print(f"sorted({words}, key=len) = {sorted_by_len}")

users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 20},
]

sorted_by_age = sorted(users, key=lambda u: u["age"])
print(f"按年龄排序: {[u['name'] for u in sorted_by_age]}")

# =============================================================================
# 7. 函数组合
# =============================================================================
print("\n=== 函数组合 ===")

def compose(*funcs):
    """函数组合"""
    def composed(x):
        for func in reversed(funcs):
            x = func(x)
        return x
    return composed

add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x**2

transform = compose(square, double, add_one)
result = transform(5)
print(f"compose(square, double, add_one)(5) = {result}")

print("\n=== 运行完成 ===")

