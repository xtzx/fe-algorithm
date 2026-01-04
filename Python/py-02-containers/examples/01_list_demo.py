#!/usr/bin/env python3
"""
01_list_demo.py - 列表操作演示
"""

# =============================================================================
# 1. 创建列表
# =============================================================================
print("=== 创建列表 ===")

lst1 = [1, 2, 3, 4, 5]
lst2 = list(range(5))
lst3 = [0] * 5
lst4 = [x**2 for x in range(5)]

print(f"字面量: {lst1}")
print(f"range: {lst2}")
print(f"重复: {lst3}")
print(f"推导式: {lst4}")

# =============================================================================
# 2. 基本操作
# =============================================================================
print("\n=== 基本操作 ===")

lst = [1, 2, 3]
lst.append(4)
print(f"append(4): {lst}")

lst.extend([5, 6])
print(f"extend([5, 6]): {lst}")

lst.insert(0, 0)
print(f"insert(0, 0): {lst}")

last = lst.pop()
print(f"pop(): 返回 {last}, 列表 {lst}")

lst.remove(3)
print(f"remove(3): {lst}")

# =============================================================================
# 3. 切片
# =============================================================================
print("\n=== 切片 ===")

lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"原列表: {lst}")
print(f"lst[2:5]: {lst[2:5]}")
print(f"lst[:5]: {lst[:5]}")
print(f"lst[5:]: {lst[5:]}")
print(f"lst[::2]: {lst[::2]}")
print(f"lst[::-1]: {lst[::-1]}")
print(f"lst[-3:]: {lst[-3:]}")

# =============================================================================
# 4. 排序
# =============================================================================
print("\n=== 排序 ===")

numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"原列表: {numbers}")

# sorted 返回新列表
sorted_nums = sorted(numbers)
print(f"sorted(): {sorted_nums}")
print(f"原列表不变: {numbers}")

# sort 原地排序
numbers.sort()
print(f"sort(): {numbers}")

# 自定义排序
words = ["apple", "pie", "banana", "cherry"]
print(f"\n按长度排序: {sorted(words, key=len)}")
print(f"降序: {sorted(words, reverse=True)}")

# =============================================================================
# 5. 查找
# =============================================================================
print("\n=== 查找 ===")

lst = [1, 2, 3, 2, 4, 2]
print(f"列表: {lst}")
print(f"2 in lst: {2 in lst}")
print(f"10 in lst: {10 in lst}")
print(f"lst.index(2): {lst.index(2)}")
print(f"lst.count(2): {lst.count(2)}")

# =============================================================================
# 6. 复制
# =============================================================================
print("\n=== 复制 ===")

import copy

original = [1, 2, [3, 4]]
shallow = original.copy()
deep = copy.deepcopy(original)

shallow[2][0] = 100
print(f"浅拷贝修改后:")
print(f"  original: {original}")  # 被影响
print(f"  shallow: {shallow}")

original = [1, 2, [3, 4]]
deep = copy.deepcopy(original)
deep[2][0] = 100
print(f"\n深拷贝修改后:")
print(f"  original: {original}")  # 不受影响
print(f"  deep: {deep}")

print("\n=== 运行完成 ===")

