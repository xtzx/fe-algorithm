#!/usr/bin/env python3
"""
02_tuple_demo.py - 元组操作演示
"""

from collections import namedtuple
from typing import NamedTuple

# =============================================================================
# 1. 创建元组
# =============================================================================
print("=== 创建元组 ===")

t1 = (1, 2, 3)
t2 = 1, 2, 3
t3 = tuple([1, 2, 3])
t_single = (1,)  # 单元素必须有逗号

print(f"括号: {t1}")
print(f"无括号: {t2}")
print(f"tuple(): {t3}")
print(f"单元素: {t_single}, 类型: {type(t_single)}")

# 单元素陷阱
not_tuple = (1)
print(f"无逗号: {not_tuple}, 类型: {type(not_tuple)}")

# =============================================================================
# 2. 不可变性
# =============================================================================
print("\n=== 不可变性 ===")

t = (1, 2, 3)
try:
    t[0] = 100
except TypeError as e:
    print(f"修改元组报错: {e}")

# 但嵌套的可变对象可以修改
t_nested = (1, 2, [3, 4])
t_nested[2][0] = 100
print(f"嵌套列表被修改: {t_nested}")

# =============================================================================
# 3. 元组解包
# =============================================================================
print("\n=== 元组解包 ===")

# 基本解包
a, b, c = (1, 2, 3)
print(f"a={a}, b={b}, c={c}")

# 交换变量
x, y = 10, 20
x, y = y, x
print(f"交换后: x={x}, y={y}")

# 星号解包
first, *rest = (1, 2, 3, 4, 5)
print(f"first={first}, rest={rest}")

*head, last = (1, 2, 3, 4, 5)
print(f"head={head}, last={last}")

first, *middle, last = (1, 2, 3, 4, 5)
print(f"first={first}, middle={middle}, last={last}")

# 忽略某些值
x, _, z = (1, 2, 3)
print(f"忽略中间: x={x}, z={z}")

# =============================================================================
# 4. 命名元组
# =============================================================================
print("\n=== 命名元组 ===")

# namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(f"Point: {p}")
print(f"p.x={p.x}, p.y={p.y}")
print(f"p[0]={p[0]}, p[1]={p[1]}")

# 解包
x, y = p
print(f"解包: x={x}, y={y}")

# typing.NamedTuple
class Person(NamedTuple):
    name: str
    age: int
    city: str = "Unknown"

person = Person("Alice", 25)
print(f"\nPerson: {person}")
print(f"name={person.name}, age={person.age}, city={person.city}")

# =============================================================================
# 5. 元组作为字典键
# =============================================================================
print("\n=== 元组作为字典键 ===")

locations = {
    (0, 0): "Origin",
    (1, 0): "East",
    (0, 1): "North",
}

print(f"(0, 0) -> {locations[(0, 0)]}")
print(f"(1, 0) -> {locations[(1, 0)]}")

# 列表不能作为键
try:
    d = {[0, 0]: "Origin"}
except TypeError as e:
    print(f"列表作为键报错: {e}")

# =============================================================================
# 6. 函数返回多值
# =============================================================================
print("\n=== 函数返回多值 ===")

def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

nums = [1, 2, 3, 4, 5]
min_val, max_val, avg = get_stats(nums)
print(f"min={min_val}, max={max_val}, avg={avg}")

# 也可以作为元组接收
result = get_stats(nums)
print(f"元组: {result}")

print("\n=== 运行完成 ===")

