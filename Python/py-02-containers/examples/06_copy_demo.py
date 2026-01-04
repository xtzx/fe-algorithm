#!/usr/bin/env python3
"""
06_copy_demo.py - 浅拷贝与深拷贝演示
"""

import copy

# =============================================================================
# 1. 引用 vs 拷贝
# =============================================================================
print("=== 引用 vs 拷贝 ===")

original = [1, 2, 3]
reference = original  # 引用同一个对象
copied = original.copy()  # 新对象

reference.append(4)
print(f"原列表: {original}")  # [1, 2, 3, 4]
print(f"引用: {reference}")   # [1, 2, 3, 4]
print(f"拷贝: {copied}")      # [1, 2, 3]

print(f"\n原列表 is 引用: {original is reference}")  # True
print(f"原列表 is 拷贝: {original is copied}")      # False

# =============================================================================
# 2. 浅拷贝
# =============================================================================
print("\n=== 浅拷贝 ===")

original = [1, 2, [3, 4]]

# 浅拷贝方法
shallow1 = original.copy()
shallow2 = list(original)
shallow3 = original[:]
shallow4 = copy.copy(original)

print(f"原列表: {original}")
print(f"浅拷贝: {shallow1}")

# 修改嵌套对象
shallow1[2][0] = 100
print(f"\n修改 shallow1[2][0] = 100 后:")
print(f"原列表: {original}")  # [1, 2, [100, 4]] - 被影响！
print(f"浅拷贝: {shallow1}")  # [1, 2, [100, 4]]

# 修改顶层对象
original = [1, 2, [3, 4]]
shallow = original.copy()
shallow[0] = 100
print(f"\n修改 shallow[0] = 100 后:")
print(f"原列表: {original}")  # [1, 2, [3, 4]] - 不受影响
print(f"浅拷贝: {shallow}")   # [100, 2, [3, 4]]

# =============================================================================
# 3. 深拷贝
# =============================================================================
print("\n=== 深拷贝 ===")

original = [1, 2, [3, 4]]
deep = copy.deepcopy(original)

print(f"原列表: {original}")
print(f"深拷贝: {deep}")

deep[2][0] = 100
print(f"\n修改 deep[2][0] = 100 后:")
print(f"原列表: {original}")  # [1, 2, [3, 4]] - 不受影响
print(f"深拷贝: {deep}")      # [1, 2, [100, 4]]

# =============================================================================
# 4. 字典的拷贝
# =============================================================================
print("\n=== 字典的拷贝 ===")

original = {"a": 1, "b": [2, 3]}

shallow = original.copy()
shallow["b"][0] = 100
print(f"浅拷贝后原字典: {original}")  # {"a": 1, "b": [100, 3]}

original = {"a": 1, "b": [2, 3]}
deep = copy.deepcopy(original)
deep["b"][0] = 100
print(f"深拷贝后原字典: {original}")  # {"a": 1, "b": [2, 3]}

# =============================================================================
# 5. 常见陷阱
# =============================================================================
print("\n=== 常见陷阱 ===")

# 陷阱 1：列表乘法创建引用
print("列表乘法陷阱:")
matrix = [[0] * 3] * 3
matrix[0][0] = 1
print(f"  [[0]*3]*3 修改后: {matrix}")  # 三行都变了！

# 正确做法
matrix = [[0] * 3 for _ in range(3)]
matrix[0][0] = 1
print(f"  推导式创建: {matrix}")  # 只有第一行变

# 陷阱 2：可变默认参数
print("\n可变默认参数陷阱:")

def bad_append(item, lst=[]):
    lst.append(item)
    return lst

print(f"  第一次调用: {bad_append(1)}")  # [1]
print(f"  第二次调用: {bad_append(2)}")  # [1, 2] - 意外！

def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(f"  正确第一次: {good_append(1)}")  # [1]
print(f"  正确第二次: {good_append(2)}")  # [2]

# =============================================================================
# 6. 何时使用深拷贝
# =============================================================================
print("\n=== 何时使用深拷贝 ===")

# 需要深拷贝的场景
print("需要深拷贝:")
print("  - 嵌套的列表/字典")
print("  - 包含可变对象的数据结构")
print("  - 需要完全独立的副本")

# 不需要深拷贝的场景
print("\n不需要深拷贝:")
print("  - 只包含不可变元素（数字、字符串）")
print("  - 只修改顶层元素")
print("  - 性能敏感的场景（深拷贝较慢）")

print("\n=== 运行完成 ===")

