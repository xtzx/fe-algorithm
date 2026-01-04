#!/usr/bin/env python3
"""
03_dict_demo.py - 字典操作演示
"""

from collections import defaultdict, Counter

# =============================================================================
# 1. 创建字典
# =============================================================================
print("=== 创建字典 ===")

d1 = {"name": "Alice", "age": 25}
d2 = dict(name="Bob", age=30)
d3 = dict([("name", "Charlie"), ("age", 35)])
d4 = {x: x**2 for x in range(5)}

print(f"字面量: {d1}")
print(f"dict(): {d2}")
print(f"键值对列表: {d3}")
print(f"推导式: {d4}")

# =============================================================================
# 2. 访问与修改
# =============================================================================
print("\n=== 访问与修改 ===")

d = {"name": "Alice", "age": 25}

# 访问
print(f"d['name']: {d['name']}")
print(f"d.get('age'): {d.get('age')}")
print(f"d.get('city'): {d.get('city')}")
print(f"d.get('city', 'N/A'): {d.get('city', 'N/A')}")

# 修改
d["age"] = 26
d["city"] = "NYC"
print(f"修改后: {d}")

# setdefault
d.setdefault("country", "USA")
print(f"setdefault: {d}")

# =============================================================================
# 3. 遍历
# =============================================================================
print("\n=== 遍历 ===")

d = {"a": 1, "b": 2, "c": 3}

print("遍历键:")
for k in d:
    print(f"  {k}")

print("遍历值:")
for v in d.values():
    print(f"  {v}")

print("遍历键值对:")
for k, v in d.items():
    print(f"  {k}: {v}")

# =============================================================================
# 4. 合并字典
# =============================================================================
print("\n=== 合并字典 ===")

d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}

# 解包方式
merged = {**d1, **d2}
print(f"{{**d1, **d2}}: {merged}")

# | 运算符（Python 3.9+）
import sys
if sys.version_info >= (3, 9):
    merged = d1 | d2
    print(f"d1 | d2: {merged}")

# update 方式
d1_copy = d1.copy()
d1_copy.update(d2)
print(f"update: {d1_copy}")

# =============================================================================
# 5. 字典推导式
# =============================================================================
print("\n=== 字典推导式 ===")

# 平方映射
squares = {x: x**2 for x in range(5)}
print(f"平方: {squares}")

# 条件过滤
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"偶数平方: {even_squares}")

# 键值互换
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(f"键值互换: {inverted}")

# =============================================================================
# 6. defaultdict
# =============================================================================
print("\n=== defaultdict ===")

# 计数
counter = defaultdict(int)
for char in "hello":
    counter[char] += 1
print(f"字符计数: {dict(counter)}")

# 分组
groups = defaultdict(list)
data = [("A", 1), ("B", 2), ("A", 3), ("B", 4)]
for key, value in data:
    groups[key].append(value)
print(f"分组: {dict(groups)}")

# =============================================================================
# 7. Counter
# =============================================================================
print("\n=== Counter ===")

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(f"计数: {counter}")
print(f"最常见的 2 个: {counter.most_common(2)}")

# 字符统计
char_counter = Counter("hello world")
print(f"字符统计: {char_counter}")

print("\n=== 运行完成 ===")

