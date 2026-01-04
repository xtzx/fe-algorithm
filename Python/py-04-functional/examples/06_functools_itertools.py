#!/usr/bin/env python3
"""
06_functools_itertools.py - functools 和 itertools 演示
"""

from functools import partial, lru_cache, reduce
from itertools import count, cycle, repeat, chain, combinations, groupby

# =============================================================================
# 1. functools.partial
# =============================================================================
print("=== functools.partial ===")

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(f"square(5) = {square(5)}")
print(f"cube(3) = {cube(3)}")

# =============================================================================
# 2. functools.lru_cache
# =============================================================================
print("\n=== functools.lru_cache ===")

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(30)
print(f"fibonacci(30) = {result}")
print(f"缓存信息: {fibonacci.cache_info()}")

# =============================================================================
# 3. itertools.count
# =============================================================================
print("\n=== itertools.count ===")

counter = count(10, 2)
first_5 = [next(counter) for _ in range(5)]
print(f"count(10, 2) 前5个: {first_5}")

# =============================================================================
# 4. itertools.combinations
# =============================================================================
print("\n=== itertools.combinations ===")

items = ["A", "B", "C"]
combs = list(combinations(items, 2))
print(f"combinations({items}, 2) = {combs}")

# =============================================================================
# 5. itertools.groupby
# =============================================================================
print("\n=== itertools.groupby ===")

words = ["apple", "pie", "banana", "cat", "dog"]
words_sorted = sorted(words, key=len)

for length, group in groupby(words_sorted, key=len):
    print(f"长度 {length}: {list(group)}")

# =============================================================================
# 6. itertools.chain
# =============================================================================
print("\n=== itertools.chain ===")

list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list(chain(list1, list2))
print(f"chain({list1}, {list2}) = {combined}")

print("\n=== 运行完成 ===")

