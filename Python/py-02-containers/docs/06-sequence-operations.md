# 06. 序列操作

## 🎯 本节目标

- 掌握通用序列操作
- 熟练使用 zip、enumerate、map、filter
- 理解迭代器协议

---

## 📏 通用操作

这些操作适用于所有序列类型（list、tuple、str 等）。

### 长度与统计

```python
lst = [3, 1, 4, 1, 5, 9, 2, 6]

len(lst)              # 8
min(lst)              # 1
max(lst)              # 9
sum(lst)              # 31

# 字符串
min("hello")          # 'e'（按字母顺序）
max("hello")          # 'o'
```

### 排序与反转

```python
lst = [3, 1, 4, 1, 5]

# sorted：返回新列表
sorted(lst)           # [1, 1, 3, 4, 5]
sorted(lst, reverse=True)  # [5, 4, 3, 1, 1]

# reversed：返回迭代器
list(reversed(lst))   # [5, 1, 4, 1, 3]

# 自定义排序
words = ["apple", "pie", "banana"]
sorted(words, key=len)           # ['pie', 'apple', 'banana']
sorted(words, key=str.lower)     # 忽略大小写
```

### 成员与索引

```python
lst = [1, 2, 3, 4, 5]

# in / not in
3 in lst              # True
10 not in lst         # True

# index
lst.index(3)          # 2
lst.index(3, 2)       # 从索引 2 开始找

# count
lst.count(1)          # 1
```

---

## 🔄 enumerate

为可迭代对象添加索引。

```python
fruits = ["apple", "banana", "cherry"]

# 基本用法
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# 指定起始索引
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")
# 1: apple
# 2: banana
# 3: cherry

# 创建带索引的数据
indexed = list(enumerate(fruits))
# [(0, 'apple'), (1, 'banana'), (2, 'cherry')]
```

### JS 对照

```javascript
// JS 中需要用 forEach 或 entries
fruits.forEach((fruit, i) => console.log(`${i}: ${fruit}`));

// 或
for (const [i, fruit] of fruits.entries()) {
    console.log(`${i}: ${fruit}`);
}
```

---

## 🔗 zip

并行遍历多个可迭代对象。

```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

# 基本用法
for name, age in zip(names, ages):
    print(f"{name} is {age}")
# Alice is 25
# Bob is 30
# Charlie is 35

# 创建字典
d = dict(zip(names, ages))
# {"Alice": 25, "Bob": 30, "Charlie": 35}

# 创建元组列表
pairs = list(zip(names, ages))
# [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# 长度不一致时截断到最短
a = [1, 2, 3]
b = [4, 5]
list(zip(a, b))  # [(1, 4), (2, 5)]

# 使用 zip_longest 保留所有
from itertools import zip_longest
list(zip_longest(a, b, fillvalue=0))
# [(1, 4), (2, 5), (3, 0)]
```

### 解压（unzip）

```python
pairs = [("a", 1), ("b", 2), ("c", 3)]

# 解压
keys, values = zip(*pairs)
# keys = ('a', 'b', 'c')
# values = (1, 2, 3)
```

---

## 🗺️ map

对每个元素应用函数。

```python
numbers = [1, 2, 3, 4, 5]

# 基本用法
squares = map(lambda x: x**2, numbers)
print(list(squares))  # [1, 4, 9, 16, 25]

# 使用普通函数
def double(x):
    return x * 2

doubled = list(map(double, numbers))

# 多个可迭代对象
a = [1, 2, 3]
b = [4, 5, 6]
sums = list(map(lambda x, y: x + y, a, b))
# [5, 7, 9]
```

### map vs 推导式

```python
# map 方式
squares = list(map(lambda x: x**2, numbers))

# 推导式（更 Pythonic）
squares = [x**2 for x in numbers]
```

---

## 🔍 filter

过滤元素。

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 基本用法
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [2, 4, 6, 8, 10]

# 过滤 Falsy 值
data = [0, 1, "", "hello", None, [], [1, 2]]
clean = list(filter(None, data))
# [1, 'hello', [1, 2]]
```

### filter vs 推导式

```python
# filter 方式
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 推导式（更 Pythonic）
evens = [x for x in numbers if x % 2 == 0]
```

---

## ✅ any 和 all

```python
numbers = [1, 2, 3, 4, 5]

# any：任意一个为真
any(x > 3 for x in numbers)   # True
any(x > 10 for x in numbers)  # False

# all：所有都为真
all(x > 0 for x in numbers)   # True
all(x > 3 for x in numbers)   # False

# 实际应用
users = [{"name": "Alice", "active": True}, {"name": "Bob", "active": False}]

# 检查是否有活跃用户
any(u["active"] for u in users)  # True

# 检查是否全部活跃
all(u["active"] for u in users)  # False
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `any(cond for x in arr)` | `arr.some(x => cond)` |
| `all(cond for x in arr)` | `arr.every(x => cond)` |

---

## 📊 reduce

累积操作（需要从 functools 导入）。

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda acc, x: acc + x, numbers)
# 15

# 求积
product = reduce(lambda acc, x: acc * x, numbers)
# 120

# 带初始值
total = reduce(lambda acc, x: acc + x, numbers, 10)
# 25
```

### JS 对照

```javascript
// JS reduce
const total = numbers.reduce((acc, x) => acc + x, 0);
```

---

## 🔧 其他实用函数

### sorted with key

```python
# 复杂排序
users = [
    {"name": "Charlie", "age": 35},
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
]

# 按年龄排序
by_age = sorted(users, key=lambda u: u["age"])

# 按多个字段排序
from operator import itemgetter
by_age_name = sorted(users, key=itemgetter("age", "name"))
```

### itertools 常用

```python
from itertools import chain, groupby, islice

# chain：连接多个可迭代对象
list(chain([1, 2], [3, 4], [5, 6]))
# [1, 2, 3, 4, 5, 6]

# groupby：分组（需要先排序）
data = [("a", 1), ("a", 2), ("b", 3)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# a [('a', 1), ('a', 2)]
# b [('b', 3)]

# islice：切片迭代器
gen = (x**2 for x in range(100))
list(islice(gen, 5))  # [0, 1, 4, 9, 16]
```

---

## ✅ 本节要点

1. `len`, `min`, `max`, `sum`, `sorted` 适用于所有序列
2. `enumerate` 添加索引
3. `zip` 并行遍历
4. `map` 映射，`filter` 过滤
5. `any` 任一为真，`all` 全部为真
6. 推导式通常比 map/filter 更 Pythonic
7. `itertools` 提供更多高级迭代工具

