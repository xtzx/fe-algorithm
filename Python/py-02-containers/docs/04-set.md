# 04. 集合 set

## 🎯 本节目标

- 掌握集合的创建与操作
- 熟练使用集合运算
- 理解 frozenset

---

## 📝 创建集合

```python
# 字面量
s = {1, 2, 3}

# ⚠️ 空集合必须用 set()
s = set()             # ✅ 空集合
s = {}                # ❌ 这是空字典！

# set() 构造函数
s = set([1, 2, 2, 3]) # {1, 2, 3}（自动去重）
s = set("hello")      # {'h', 'e', 'l', 'o'}

# 集合推导式
s = {x**2 for x in range(5)}  # {0, 1, 4, 9, 16}
```

### 集合的特点

1. **无序**：元素没有固定顺序
2. **唯一**：自动去重
3. **元素必须可哈希**：不能包含列表、字典等

```python
# 元素必须可哈希
s = {1, "hello", (1, 2)}  # ✅
s = {1, [2, 3]}           # ❌ TypeError
```

---

## 🔧 基本操作

### 添加元素

```python
s = {1, 2, 3}

# add：添加单个元素
s.add(4)              # {1, 2, 3, 4}
s.add(3)              # {1, 2, 3, 4}（已存在，无效果）

# update：添加多个元素
s.update([5, 6])      # {1, 2, 3, 4, 5, 6}
s.update({7, 8})
s.update("ab")        # 添加 'a' 和 'b'
```

### 删除元素

```python
s = {1, 2, 3, 4, 5}

# remove：删除指定元素（不存在报错）
s.remove(5)           # {1, 2, 3, 4}
s.remove(10)          # ❌ KeyError

# discard：删除指定元素（不存在不报错）
s.discard(4)          # {1, 2, 3}
s.discard(10)         # 无效果，不报错

# pop：删除并返回任意元素
val = s.pop()         # 返回某个元素

# clear：清空
s.clear()             # set()
```

### 查找

```python
s = {1, 2, 3}

# in：检查存在
2 in s                # True
5 in s                # False

# 长度
len(s)                # 3
```

### JS 对照表

| Python | JavaScript |
|--------|------------|
| `s.add(x)` | `set.add(x)` |
| `s.remove(x)` | `set.delete(x)` |
| `s.discard(x)` | `set.delete(x)` |
| `x in s` | `set.has(x)` |
| `len(s)` | `set.size` |
| `s.clear()` | `set.clear()` |

---

## ➕ 集合运算

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
```

### 并集（Union）

```python
a | b                 # {1, 2, 3, 4, 5, 6}
a.union(b)            # 同上

# 原地修改
a |= b
a.update(b)
```

### 交集（Intersection）

```python
a & b                 # {3, 4}
a.intersection(b)     # 同上

# 原地修改
a &= b
a.intersection_update(b)
```

### 差集（Difference）

```python
a - b                 # {1, 2}（在 a 中但不在 b 中）
a.difference(b)       # 同上

b - a                 # {5, 6}

# 原地修改
a -= b
a.difference_update(b)
```

### 对称差集（Symmetric Difference）

```python
a ^ b                 # {1, 2, 5, 6}（不同时在两者中）
a.symmetric_difference(b)

# 原地修改
a ^= b
a.symmetric_difference_update(b)
```

### 集合关系

```python
a = {1, 2}
b = {1, 2, 3, 4}

# 子集
a <= b                # True
a.issubset(b)         # True
a < b                 # True（真子集）

# 超集
b >= a                # True
b.issuperset(a)       # True
b > a                 # True（真超集）

# 不相交
a.isdisjoint({5, 6})  # True（无共同元素）
```

---

## 🔐 frozenset

不可变集合，可以作为字典键或集合元素。

```python
# 创建
fs = frozenset([1, 2, 3])

# 不能修改
fs.add(4)             # ❌ AttributeError

# 支持集合运算（返回新 frozenset）
fs2 = frozenset([3, 4, 5])
fs | fs2              # frozenset({1, 2, 3, 4, 5})
fs & fs2              # frozenset({3})

# 可以作为字典键
d = {frozenset({1, 2}): "value"}

# 可以作为集合元素
s = {frozenset({1}), frozenset({2})}
```

---

## 🎯 实际应用

### 去重

```python
# 列表去重（不保序）
lst = [1, 2, 2, 3, 3, 3]
unique = list(set(lst))  # [1, 2, 3]

# 去重并保持顺序
unique = list(dict.fromkeys(lst))  # [1, 2, 3]
```

### 查找共同/不同元素

```python
users_a = {"alice", "bob", "charlie"}
users_b = {"bob", "david", "eve"}

# 共同用户
common = users_a & users_b  # {"bob"}

# 只在 A 的用户
only_a = users_a - users_b  # {"alice", "charlie"}

# 所有用户
all_users = users_a | users_b
```

### 成员检测（比列表快）

```python
# 需要频繁检测成员
valid_ids = {1, 2, 3, 4, 5}  # 用 set，O(1)

if user_id in valid_ids:
    print("Valid")

# 不要用列表
valid_ids = [1, 2, 3, 4, 5]  # O(n)
```

### 过滤

```python
all_items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
blacklist = {2, 4, 6, 8}

filtered = [x for x in all_items if x not in blacklist]
# [1, 3, 5, 7, 9]
```

---

## ⚠️ 常见坑

### 空集合

```python
# ❌ 错误
s = {}
print(type(s))  # <class 'dict'>

# ✅ 正确
s = set()
print(type(s))  # <class 'set'>
```

### 集合是无序的

```python
s = {3, 1, 2}
list(s)  # 顺序不确定！可能是 [1, 2, 3] 或其他

# 需要有序时先排序
sorted(s)  # [1, 2, 3]
```

### 元素必须可哈希

```python
# ❌ 不能包含列表
s = {[1, 2]}  # TypeError

# ✅ 用元组代替
s = {(1, 2)}
```

---

## ✅ 本节要点

1. 空集合用 `set()`，`{}` 是空字典
2. 集合元素必须可哈希
3. `|` 并集，`&` 交集，`-` 差集，`^` 对称差集
4. `frozenset` 不可变，可作为字典键
5. 成员检测用集合比列表快
6. 集合是无序的

