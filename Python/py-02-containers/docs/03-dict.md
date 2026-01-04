# 03. 字典 dict

## 🎯 本节目标

- 掌握字典的创建与操作
- 熟练遍历字典
- 了解 defaultdict 和 Counter

---

## 📝 创建字典

```python
# 字面量
d = {"name": "Alice", "age": 25}

# dict() 构造函数
d = dict()                         # 空字典
d = dict(name="Alice", age=25)     # 关键字参数
d = dict([("name", "Alice"), ("age", 25)])  # 键值对列表

# 字典推导式
d = {x: x**2 for x in range(5)}   # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# fromkeys：统一值
d = dict.fromkeys(["a", "b", "c"], 0)  # {"a": 0, "b": 0, "c": 0}
```

### JS 对照

```javascript
// JS 创建对象
const obj = {name: "Alice", age: 25};
const obj = Object.fromEntries([["name", "Alice"]]);

// JS Map
const map = new Map([["name", "Alice"]]);
```

---

## 🔧 基本操作

### 访问与修改

```python
d = {"name": "Alice", "age": 25}

# 访问
d["name"]             # "Alice"
d["city"]             # ❌ KeyError!

# 安全访问
d.get("name")         # "Alice"
d.get("city")         # None
d.get("city", "N/A")  # "N/A"（默认值）

# 修改/添加
d["age"] = 26
d["city"] = "NYC"

# setdefault：不存在则设置
d.setdefault("country", "USA")  # 设置并返回 "USA"
d.setdefault("name", "Bob")     # 已存在，返回 "Alice"
```

### 删除

```python
d = {"a": 1, "b": 2, "c": 3}

# pop：删除并返回
val = d.pop("a")      # 1
val = d.pop("x", 0)   # 不存在返回默认值 0

# del
del d["b"]

# popitem：删除并返回最后一个（Python 3.7+）
key, val = d.popitem()

# clear
d.clear()
```

### 合并字典

```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}

# 方式 1：update（原地修改）
d1.update(d2)         # d1 = {"a": 1, "b": 3, "c": 4}

# 方式 2：| 运算符（Python 3.9+）
d3 = d1 | d2          # 新字典

# 方式 3：解包
d3 = {**d1, **d2}
```

### JS 对照表

| Python | JavaScript |
|--------|------------|
| `d["key"]` | `obj.key` 或 `obj["key"]` |
| `d.get("key")` | `obj.key ?? undefined` |
| `d.get("key", default)` | `obj.key ?? default` |
| `"key" in d` | `"key" in obj` |
| `del d["key"]` | `delete obj.key` |
| `d.update(d2)` | `Object.assign(obj, obj2)` |
| `{**d1, **d2}` | `{...obj1, ...obj2}` |

---

## 🔄 遍历字典

```python
d = {"a": 1, "b": 2, "c": 3}

# 遍历键
for key in d:
    print(key)

for key in d.keys():
    print(key)

# 遍历值
for value in d.values():
    print(value)

# 遍历键值对（推荐）
for key, value in d.items():
    print(f"{key}: {value}")
```

### 视图对象

```python
d = {"a": 1, "b": 2}

keys = d.keys()       # dict_keys(['a', 'b'])
values = d.values()   # dict_values([1, 2])
items = d.items()     # dict_items([('a', 1), ('b', 2)])

# 视图是动态的
d["c"] = 3
print(keys)           # dict_keys(['a', 'b', 'c'])

# 转为列表
key_list = list(d.keys())
```

---

## 📊 字典推导式

```python
# 基本
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 条件过滤
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# 从两个列表创建
keys = ["a", "b", "c"]
values = [1, 2, 3]
d = {k: v for k, v in zip(keys, values)}
# {"a": 1, "b": 2, "c": 3}

# 键值互换
original = {"a": 1, "b": 2}
reversed_d = {v: k for k, v in original.items()}
# {1: "a", 2: "b"}
```

---

## 🔧 defaultdict

```python
from collections import defaultdict

# 默认值为 int（即 0）
counter = defaultdict(int)
for word in ["a", "b", "a", "c", "a"]:
    counter[word] += 1
# {'a': 3, 'b': 1, 'c': 1}

# 默认值为 list
groups = defaultdict(list)
for name, category in [("Alice", "A"), ("Bob", "B"), ("Charlie", "A")]:
    groups[category].append(name)
# {'A': ['Alice', 'Charlie'], 'B': ['Bob']}

# 默认值为自定义函数
d = defaultdict(lambda: "N/A")
print(d["missing"])  # "N/A"
```

### 对比普通 dict

```python
# 普通 dict
d = {}
for word in words:
    if word not in d:
        d[word] = 0
    d[word] += 1

# 或使用 setdefault
d = {}
for word in words:
    d.setdefault(word, 0)
    d[word] += 1

# defaultdict 更简洁
d = defaultdict(int)
for word in words:
    d[word] += 1
```

---

## 📊 Counter

```python
from collections import Counter

# 创建
c = Counter(["a", "b", "a", "c", "a", "b"])
# Counter({'a': 3, 'b': 2, 'c': 1})

c = Counter("hello")
# Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# 常用方法
c.most_common(2)      # [('a', 3), ('b', 2)]
c["a"]                # 3
c["x"]                # 0（不存在返回 0，不报错）

# 更新
c.update(["a", "d"])  # 增加计数
c.subtract(["a"])     # 减少计数

# 运算
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
c1 + c2               # Counter({'a': 4, 'b': 3})
c1 - c2               # Counter({'a': 2})（只保留正数）
c1 & c2               # Counter({'a': 1, 'b': 1})（取最小）
c1 | c2               # Counter({'a': 3, 'b': 2})（取最大）
```

---

## 📋 字典有序性

```python
# Python 3.7+ 字典保持插入顺序
d = {}
d["first"] = 1
d["second"] = 2
d["third"] = 3

list(d.keys())  # ['first', 'second', 'third']

# OrderedDict（3.7 前的有序字典，现在基本不需要了）
from collections import OrderedDict
```

---

## 🔑 字典键的要求

键必须是**可哈希的**（hashable）：

```python
# ✅ 可以作为键
d[1] = "int"
d["key"] = "str"
d[(1, 2)] = "tuple"
d[frozenset({1, 2})] = "frozenset"

# ❌ 不能作为键
d[[1, 2]] = "list"     # TypeError: unhashable type: 'list'
d[{1, 2}] = "set"      # TypeError: unhashable type: 'set'
d[{"a": 1}] = "dict"   # TypeError: unhashable type: 'dict'
```

**可哈希** = 不可变 + 有 `__hash__` 方法

---

## ✅ 本节要点

1. `d.get(key, default)` 安全访问
2. `d.setdefault(key, value)` 不存在则设置
3. `for k, v in d.items()` 遍历键值对
4. `{**d1, **d2}` 或 `d1 | d2` 合并字典
5. `defaultdict` 自动初始化默认值
6. `Counter` 快速统计计数
7. 字典键必须可哈希

