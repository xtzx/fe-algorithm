# 08. itertools 模块

## 🎯 本节目标

- 掌握 itertools 常用函数
- 处理无限序列
- 组合和排列
- 分组和过滤

---

## 📝 itertools 概述

`itertools` 提供了高效的迭代器工具函数。

```python
import itertools
```

---

## ♾️ 无限迭代器

### count：计数器

```python
from itertools import count

# 从 0 开始，步长为 1
for i, n in enumerate(count()):
    if i >= 5:
        break
    print(n)
# 0, 1, 2, 3, 4

# 指定起始值和步长
for i, n in enumerate(count(10, 2)):
    if i >= 5:
        break
    print(n)
# 10, 12, 14, 16, 18
```

### cycle：循环

```python
from itertools import cycle

colors = cycle(["red", "green", "blue"])
for i, color in enumerate(colors):
    if i >= 7:
        break
    print(color)
# red, green, blue, red, green, blue, red
```

### repeat：重复

```python
from itertools import repeat

# 无限重复
for i, value in enumerate(repeat("hello")):
    if i >= 3:
        break
    print(value)
# hello, hello, hello

# 指定次数
print(list(repeat("hello", 3)))
# ['hello', 'hello', 'hello']
```

---

## 🔗 组合迭代器

### chain：连接

```python
from itertools import chain

# 连接多个可迭代对象
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

combined = list(chain(list1, list2, list3))
print(combined)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 展平嵌套列表
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))
print(flat)  # [1, 2, 3, 4, 5, 6]
```

### zip_longest：长 zip

```python
from itertools import zip_longest

a = [1, 2, 3]
b = [4, 5]

# zip 截断到最短
print(list(zip(a, b)))  # [(1, 4), (2, 5)]

# zip_longest 填充到最长
print(list(zip_longest(a, b, fillvalue=0)))
# [(1, 4), (2, 5), (3, 0)]
```

### product：笛卡尔积

```python
from itertools import product

# 两个集合的笛卡尔积
colors = ["red", "blue"]
sizes = ["S", "M", "L"]

combinations = list(product(colors, sizes))
print(combinations)
# [('red', 'S'), ('red', 'M'), ('red', 'L'),
#  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# 多个集合
print(list(product([1, 2], [3, 4], [5, 6])))
# [(1, 3, 5), (1, 3, 6), (1, 4, 5), ...]
```

### permutations：排列

```python
from itertools import permutations

# 排列（顺序重要）
items = ["A", "B", "C"]
perms = list(permutations(items, 2))
print(perms)
# [('A', 'B'), ('A', 'C'), ('B', 'A'),
#  ('B', 'C'), ('C', 'A'), ('C', 'B')]

# 全排列
full_perms = list(permutations(items))
print(len(full_perms))  # 6
```

### combinations：组合

```python
from itertools import combinations

# 组合（顺序不重要）
items = ["A", "B", "C"]
combs = list(combinations(items, 2))
print(combs)
# [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 带重复的组合
from itertools import combinations_with_replacement
combs_repeat = list(combinations_with_replacement(items, 2))
print(combs_repeat)
# [('A', 'A'), ('A', 'B'), ('A', 'C'),
#  ('B', 'B'), ('B', 'C'), ('C', 'C')]
```

---

## 🔍 过滤迭代器

### takewhile：取满足条件的

```python
from itertools import takewhile

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 取小于 5 的元素
result = list(takewhile(lambda x: x < 5, numbers))
print(result)  # [1, 2, 3, 4]
```

### dropwhile：跳过满足条件的

```python
from itertools import dropwhile

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 跳过小于 5 的元素
result = list(dropwhile(lambda x: x < 5, numbers))
print(result)  # [5, 6, 7, 8, 9, 10]
```

### filterfalse：过滤不满足条件的

```python
from itertools import filterfalse

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 取不满足条件的（奇数）
odds = list(filterfalse(lambda x: x % 2 == 0, numbers))
print(odds)  # [1, 3, 5, 7, 9]
```

### islice：切片迭代器

```python
from itertools import islice

numbers = range(100)

# 切片迭代器（不创建列表）
first_10 = list(islice(numbers, 10))
print(first_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 指定范围
middle = list(islice(numbers, 10, 20))
print(middle)  # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# 指定步长
every_5th = list(islice(numbers, 0, 20, 5))
print(every_5th)  # [0, 5, 10, 15]
```

---

## 📊 分组迭代器

### groupby：分组

```python
from itertools import groupby

# ⚠️ 重要：需要先排序
data = [("A", 1), ("A", 2), ("B", 3), ("B", 4), ("A", 5)]
data_sorted = sorted(data, key=lambda x: x[0])

for key, group in groupby(data_sorted, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")
# A: [('A', 1), ('A', 2)]
# A: [('A', 5)]
# B: [('B', 3), ('B', 4)]
```

### 实际应用

```python
from itertools import groupby

# 按长度分组单词
words = ["apple", "pie", "banana", "cat", "dog", "cherry"]
words_sorted = sorted(words, key=len)

for length, group in groupby(words_sorted, key=len):
    print(f"长度 {length}: {list(group)}")
# 长度 3: ['pie', 'cat', 'dog']
# 长度 5: ['apple']
# 长度 6: ['banana', 'cherry']
```

---

## 🔄 其他实用函数

### accumulate：累积

```python
from itertools import accumulate

numbers = [1, 2, 3, 4, 5]

# 默认求和
sums = list(accumulate(numbers))
print(sums)  # [1, 3, 6, 10, 15]

# 自定义函数
products = list(accumulate(numbers, lambda x, y: x * y))
print(products)  # [1, 2, 6, 24, 120]
```

### tee：复制迭代器

```python
from itertools import tee

numbers = range(5)
it1, it2, it3 = tee(numbers, 3)

print(list(it1))  # [0, 1, 2, 3, 4]
print(list(it2))  # [0, 1, 2, 3, 4]
print(list(it3))  # [0, 1, 2, 3, 4]
```

### starmap：星号 map

```python
from itertools import starmap

# map 需要多个参数
pairs = [(2, 3), (4, 5), (6, 7)]
result = list(starmap(pow, pairs))
print(result)  # [8, 1024, 279936] (2**3, 4**5, 6**7)
```

---

## 🎯 实际应用

### 1. 生成所有组合

```python
from itertools import combinations

def find_combinations(items, target_sum):
    """找出所有和为目标值的组合"""
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            if sum(combo) == target_sum:
                yield combo

numbers = [1, 2, 3, 4, 5]
for combo in find_combinations(numbers, 5):
    print(combo)
# (5,)
# (1, 4)
# (2, 3)
```

### 2. 滑动窗口

```python
from itertools import islice

def sliding_window(iterable, n):
    """滑动窗口"""
    it = iter(iterable)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for x in it:
        window = window[1:] + (x,)
        yield window

numbers = [1, 2, 3, 4, 5, 6]
for window in sliding_window(numbers, 3):
    print(window)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)
# (4, 5, 6)
```

### 3. 分批处理

```python
from itertools import islice

def batched(iterable, n):
    """分批处理"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

numbers = range(10)
for batch in batched(numbers, 3):
    print(batch)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]
```

---

## ⚠️ 常见陷阱

### 1. groupby 需要排序

```python
# ❌ 错误：未排序
data = [("A", 1), ("B", 2), ("A", 3)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1)]
# B [('B', 2)]
# A [('A', 3)]  # A 被分成两组！

# ✅ 正确：先排序
data_sorted = sorted(data, key=lambda x: x[0])
for key, group in groupby(data_sorted, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1), ('A', 3)]
# B [('B', 2)]
```

### 2. 迭代器只能使用一次

```python
it = count()
list(islice(it, 5))  # [0, 1, 2, 3, 4]
list(islice(it, 5))  # [5, 6, 7, 8, 9]（继续）
```

---

## ✅ 本节要点

1. `count`, `cycle`, `repeat` 创建无限序列
2. `chain` 连接，`zip_longest` 长 zip
3. `product` 笛卡尔积，`permutations` 排列，`combinations` 组合
4. `takewhile`, `dropwhile`, `filterfalse` 过滤
5. `groupby` 分组（需要先排序）
6. `islice` 切片迭代器
7. `accumulate` 累积操作

