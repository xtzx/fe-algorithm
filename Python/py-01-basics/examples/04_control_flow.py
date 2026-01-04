#!/usr/bin/env python3
"""
04_control_flow.py - 控制流演示

演示：
- if/elif/else
- for 循环
- while 循环
- break/continue/else
- match-case（Python 3.10+）
"""

# =============================================================================
# 1. 条件语句
# =============================================================================

print("=== 条件语句 ===")

age = 25

if age < 13:
    print("儿童")
elif age < 18:
    print("青少年")
elif age < 60:
    print("成年人")
else:
    print("老年人")

# 三元表达式
status = "成年" if age >= 18 else "未成年"
print(f"状态: {status}")

# 链式比较
score = 85
if 80 <= score < 90:
    print(f"分数 {score} 是 B 级")

# =============================================================================
# 2. for 循环
# =============================================================================

print("\n=== for 循环 ===")

# 遍历列表
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"水果: {fruit}")

# 遍历字符串
print("\n遍历字符串:")
for char in "Python":
    print(char, end=" ")
print()

# range()
print("\nrange(5):")
for i in range(5):
    print(i, end=" ")
print()

print("\nrange(1, 6):")
for i in range(1, 6):
    print(i, end=" ")
print()

print("\nrange(0, 10, 2):")
for i in range(0, 10, 2):
    print(i, end=" ")
print()

# enumerate()
print("\nenumerate():")
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")

# zip()
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
print("\nzip():")
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# =============================================================================
# 3. 列表推导式
# =============================================================================

print("\n=== 列表推导式 ===")

# 基本
squares = [x**2 for x in range(10)]
print(f"平方: {squares}")

# 带条件
evens = [x for x in range(20) if x % 2 == 0]
print(f"偶数: {evens}")

# 嵌套
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"展平: {flat}")

# 字典推导式
square_dict = {x: x**2 for x in range(5)}
print(f"平方字典: {square_dict}")

# =============================================================================
# 4. while 循环
# =============================================================================

print("\n=== while 循环 ===")

count = 0
while count < 5:
    print(f"count = {count}")
    count += 1

# =============================================================================
# 5. break, continue, else
# =============================================================================

print("\n=== break ===")
for i in range(10):
    if i == 5:
        break
    print(i, end=" ")
print()

print("\n=== continue ===")
for i in range(10):
    if i % 2 == 0:
        continue
    print(i, end=" ")
print()

print("\n=== 循环的 else ===")
# else 在循环正常结束时执行（未被 break 中断）
for i in range(5):
    if i == 10:  # 不会触发
        break
else:
    print("循环正常结束（未 break）")

# 查找示例
target = 7
numbers = [1, 3, 5, 7, 9]

for num in numbers:
    if num == target:
        print(f"找到 {target}")
        break
else:
    print(f"未找到 {target}")

# =============================================================================
# 6. match-case（Python 3.10+）
# =============================================================================

print("\n=== match-case（Python 3.10+）===")

# 检查 Python 版本
import sys
if sys.version_info >= (3, 10):
    command = "start"

    match command:
        case "start":
            print("启动...")
        case "stop":
            print("停止...")
        case _:
            print("未知命令")

    # 匹配序列
    point = (0, 5)
    match point:
        case (0, 0):
            print("原点")
        case (0, y):
            print(f"Y 轴上，y = {y}")
        case (x, 0):
            print(f"X 轴上，x = {x}")
        case (x, y):
            print(f"点 ({x}, {y})")
else:
    print(f"当前 Python 版本 {sys.version_info[:2]} 不支持 match-case")
    print("需要 Python 3.10+")

print("\n=== 运行完成 ===")

