#!/usr/bin/env python3
"""
05_generators.py - 生成器演示
"""

# =============================================================================
# 1. 生成器函数
# =============================================================================
print("=== 生成器函数 ===")

def countdown(n):
    """倒计时生成器"""
    while n > 0:
        yield n
        n -= 1

gen = countdown(5)
print(f"类型: {type(gen)}")
for value in gen:
    print(value, end=" ")
print()

# =============================================================================
# 2. 斐波那契生成器
# =============================================================================
print("\n=== 斐波那契生成器 ===")

def fibonacci():
    """无限斐波那契生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(f"前10个: {first_10}")

# =============================================================================
# 3. 生成器表达式
# =============================================================================
print("\n=== 生成器表达式 ===")

# 列表推导式（立即生成）
squares_list = [x**2 for x in range(10)]
print(f"列表: {squares_list[:5]}...")

# 生成器表达式（惰性生成）
squares_gen = (x**2 for x in range(10))
print(f"生成器: {type(squares_gen)}")
print(f"前5个: {[next(squares_gen) for _ in range(5)]}")

# =============================================================================
# 4. yield from
# =============================================================================
print("\n=== yield from ===")

def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    yield from generator1()
    yield from generator2()

for value in combined():
    print(value, end=" ")
print()

# =============================================================================
# 5. 展平嵌套结构
# =============================================================================
print("\n=== 展平嵌套结构 ===")

def flatten(nested):
    """展平嵌套列表"""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3], [4, [5, 6]], 7]
flat = list(flatten(nested))
print(f"展平: {flat}")

# =============================================================================
# 6. 内存效率对比
# =============================================================================
print("\n=== 内存效率对比 ===")

# 列表：占用内存
big_list = [x**2 for x in range(1000)]
print(f"列表大小: {len(big_list)}")

# 生成器：几乎不占内存
big_gen = (x**2 for x in range(1000))
print(f"生成器类型: {type(big_gen)}")

# 只使用前几个
count = 0
for value in big_gen:
    if value > 100:
        break
    count += 1
print(f"只处理了 {count} 个值")

print("\n=== 运行完成 ===")

