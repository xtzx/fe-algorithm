# 09. 练习题

> 25 道练习题，分模块覆盖容器操作

---

## 📝 List 操作（8 道）

### 1. 列表去重保序

**题目**：去除列表中的重复元素，保持原有顺序。

<details>
<summary>答案</summary>

```python
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# 或使用 dict.fromkeys（Python 3.7+）
def remove_duplicates_v2(lst):
    return list(dict.fromkeys(lst))

print(remove_duplicates([1, 2, 2, 3, 1, 4]))  # [1, 2, 3, 4]
```

</details>

---

### 2. 列表扁平化

**题目**：将嵌套列表展平为一维列表。

<details>
<summary>答案</summary>

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# 测试
nested = [1, [2, 3], [4, [5, 6]], 7]
print(flatten(nested))  # [1, 2, 3, 4, 5, 6, 7]
```

</details>

---

### 3. 列表分块

**题目**：将列表分成指定大小的块。

<details>
<summary>答案</summary>

```python
def chunk(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

print(chunk([1, 2, 3, 4, 5, 6, 7], 3))
# [[1, 2, 3], [4, 5, 6], [7]]
```

</details>

---

### 4. 列表旋转

**题目**：将列表向右旋转 k 位。

<details>
<summary>答案</summary>

```python
def rotate(lst, k):
    if not lst:
        return lst
    k = k % len(lst)
    return lst[-k:] + lst[:-k]

print(rotate([1, 2, 3, 4, 5], 2))  # [4, 5, 1, 2, 3]
```

</details>

---

### 5. 找出两个列表的交集

<details>
<summary>答案</summary>

```python
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# 保持顺序的版本
def intersection_ordered(lst1, lst2):
    set2 = set(lst2)
    return [x for x in lst1 if x in set2]

print(intersection([1, 2, 3, 4], [3, 4, 5, 6]))  # [3, 4]
```

</details>

---

### 6. 列表差集

**题目**：找出在 lst1 中但不在 lst2 中的元素。

<details>
<summary>答案</summary>

```python
def difference(lst1, lst2):
    set2 = set(lst2)
    return [x for x in lst1 if x not in set2]

print(difference([1, 2, 3, 4], [3, 4, 5]))  # [1, 2]
```

</details>

---

### 7. 合并有序列表

**题目**：合并两个有序列表，保持有序。

<details>
<summary>答案</summary>

```python
def merge_sorted(lst1, lst2):
    result = []
    i = j = 0
    while i < len(lst1) and j < len(lst2):
        if lst1[i] <= lst2[j]:
            result.append(lst1[i])
            i += 1
        else:
            result.append(lst2[j])
            j += 1
    result.extend(lst1[i:])
    result.extend(lst2[j:])
    return result

print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
```

</details>

---

### 8. 移动零到末尾

**题目**：将列表中的所有零移动到末尾，保持非零元素顺序。

<details>
<summary>答案</summary>

```python
def move_zeros(lst):
    non_zeros = [x for x in lst if x != 0]
    zeros = [0] * (len(lst) - len(non_zeros))
    return non_zeros + zeros

print(move_zeros([0, 1, 0, 3, 12]))  # [1, 3, 12, 0, 0]
```

</details>

---

## 📖 Dict 操作（8 道）

### 9. 字典值求和

**题目**：计算字典中所有数值的和。

<details>
<summary>答案</summary>

```python
def sum_values(d):
    return sum(d.values())

print(sum_values({"a": 1, "b": 2, "c": 3}))  # 6
```

</details>

---

### 10. 合并字典（值相加）

**题目**：合并两个字典，相同键的值相加。

<details>
<summary>答案</summary>

```python
def merge_add(d1, d2):
    result = d1.copy()
    for k, v in d2.items():
        result[k] = result.get(k, 0) + v
    return result

print(merge_add({"a": 1, "b": 2}, {"b": 3, "c": 4}))
# {"a": 1, "b": 5, "c": 4}
```

</details>

---

### 11. 按值排序字典

**题目**：返回按值排序后的键列表。

<details>
<summary>答案</summary>

```python
def sort_by_value(d, reverse=False):
    return sorted(d.keys(), key=lambda k: d[k], reverse=reverse)

d = {"a": 3, "b": 1, "c": 2}
print(sort_by_value(d))  # ["b", "c", "a"]
```

</details>

---

### 12. 字典反转

**题目**：交换字典的键和值。

<details>
<summary>答案</summary>

```python
def invert_dict(d):
    return {v: k for k, v in d.items()}

print(invert_dict({"a": 1, "b": 2}))  # {1: "a", 2: "b"}
```

</details>

---

### 13. 分组

**题目**：按某个键对字典列表进行分组。

<details>
<summary>答案</summary>

```python
from collections import defaultdict

def group_by(items, key):
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)

users = [
    {"name": "Alice", "dept": "A"},
    {"name": "Bob", "dept": "B"},
    {"name": "Charlie", "dept": "A"},
]
print(group_by(users, "dept"))
```

</details>

---

### 14. 嵌套字典访问

**题目**：安全地访问嵌套字典。

<details>
<summary>答案</summary>

```python
def get_nested(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

data = {"a": {"b": {"c": 1}}}
print(get_nested(data, "a", "b", "c"))  # 1
print(get_nested(data, "a", "x", "y", default=0))  # 0
```

</details>

---

### 15. 过滤字典

**题目**：过滤出值大于指定阈值的键值对。

<details>
<summary>答案</summary>

```python
def filter_dict(d, threshold):
    return {k: v for k, v in d.items() if v > threshold}

print(filter_dict({"a": 1, "b": 5, "c": 3}, 2))
# {"b": 5, "c": 3}
```

</details>

---

### 16. 词频统计

**题目**：统计字符串中每个单词的出现次数。

<details>
<summary>答案</summary>

```python
from collections import Counter

def word_count(text):
    words = text.lower().split()
    return dict(Counter(words))

print(word_count("hello world hello python"))
# {"hello": 2, "world": 1, "python": 1}
```

</details>

---

## 🎨 推导式（6 道）

### 17. 矩阵转置

<details>
<summary>答案</summary>

```python
def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

matrix = [[1, 2, 3], [4, 5, 6]]
print(transpose(matrix))  # [[1, 4], [2, 5], [3, 6]]
```

</details>

---

### 18. 生成九九乘法表

<details>
<summary>答案</summary>

```python
table = [[i * j for j in range(1, 10)] for i in range(1, 10)]
# 或生成字符串
table_str = [f"{i}*{j}={i*j}" for i in range(1, 10) for j in range(1, i+1)]
```

</details>

---

### 19. 筛选素数

<details>
<summary>答案</summary>

```python
def is_prime(n):
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

primes = [x for x in range(2, 100) if is_prime(x)]
```

</details>

---

### 20. 字符串单词长度映射

<details>
<summary>答案</summary>

```python
text = "Hello World Python Programming"
lengths = {word: len(word) for word in text.split()}
# {"Hello": 5, "World": 5, "Python": 6, "Programming": 11}
```

</details>

---

### 21. 展平嵌套字典

<details>
<summary>答案</summary>

```python
def flatten_dict(d, prefix=""):
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items

nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
print(flatten_dict(nested))
# {"a": 1, "b.c": 2, "b.d.e": 3}
```

</details>

---

### 22. 生成器：斐波那契数列

<details>
<summary>答案</summary>

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(list(fibonacci(10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

</details>

---

## 🔧 综合题（3 道）

### 23. 实现简单的 LRU 缓存

<details>
<summary>答案</summary>

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

</details>

---

### 24. 两数之和

**题目**：找出列表中两个数之和等于目标值的索引。

<details>
<summary>答案</summary>

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

</details>

---

### 25. 实现字典的 dot 访问

<details>
<summary>答案</summary>

```python
class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

d = DotDict({"name": "Alice", "age": 25})
print(d.name)  # Alice
d.city = "NYC"
print(d["city"])  # NYC
```

</details>

