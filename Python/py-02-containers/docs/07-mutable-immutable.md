# 07. 可变与不可变

## 🎯 本节目标

- 理解可变和不可变类型
- 掌握浅拷贝与深拷贝
- 避免常见陷阱

---

## 📋 类型分类

### 不可变类型（Immutable）

创建后不能修改的类型：

```python
# 数字
x = 42
x = 43      # 创建新对象，不是修改

# 字符串
s = "hello"
s[0] = "H"  # ❌ TypeError

# 元组
t = (1, 2, 3)
t[0] = 100  # ❌ TypeError

# frozenset
fs = frozenset({1, 2, 3})
fs.add(4)   # ❌ AttributeError
```

**不可变类型**：`int`, `float`, `str`, `bool`, `tuple`, `frozenset`, `bytes`

### 可变类型（Mutable）

创建后可以修改的类型：

```python
# 列表
lst = [1, 2, 3]
lst[0] = 100  # ✅ [100, 2, 3]
lst.append(4) # ✅ [100, 2, 3, 4]

# 字典
d = {"a": 1}
d["b"] = 2    # ✅ {"a": 1, "b": 2}

# 集合
s = {1, 2, 3}
s.add(4)      # ✅ {1, 2, 3, 4}
```

**可变类型**：`list`, `dict`, `set`, `bytearray`

---

## 🔑 可哈希性

### 什么是可哈希？

- 可哈希对象可以作为字典键和集合元素
- 通常不可变类型是可哈希的

```python
# ✅ 可哈希
hash(42)              # 42
hash("hello")         # 一个整数
hash((1, 2, 3))       # 元组可哈希
hash(frozenset({1}))  # frozenset 可哈希

# ❌ 不可哈希
hash([1, 2, 3])       # TypeError: unhashable type: 'list'
hash({1, 2, 3})       # TypeError: unhashable type: 'set'
hash({"a": 1})        # TypeError: unhashable type: 'dict'
```

### 字典键的要求

```python
d = {}

# ✅ 可以作为键
d[1] = "int"
d["key"] = "str"
d[(1, 2)] = "tuple"

# ❌ 不能作为键
d[[1, 2]] = "list"    # TypeError
d[{1, 2}] = "set"     # TypeError
```

### 特殊情况：包含可变元素的元组

```python
# 元组包含列表
t = (1, [2, 3])
hash(t)               # ❌ TypeError

# 只有元素都可哈希时，元组才可哈希
t = (1, (2, 3))
hash(t)               # ✅
```

---

## 📋 函数参数的行为

### 不可变参数

```python
def modify_int(x):
    x = 100
    return x

a = 42
result = modify_int(a)
print(a)      # 42（未变）
print(result) # 100
```

### 可变参数

```python
def modify_list(lst):
    lst.append(100)
    return lst

a = [1, 2, 3]
result = modify_list(a)
print(a)      # [1, 2, 3, 100]（被修改了！）
print(result) # [1, 2, 3, 100]
```

### 避免意外修改

```python
def safe_modify(lst):
    lst = lst.copy()  # 创建副本
    lst.append(100)
    return lst

a = [1, 2, 3]
result = safe_modify(a)
print(a)      # [1, 2, 3]（未变）
print(result) # [1, 2, 3, 100]
```

---

## 🔄 浅拷贝 vs 深拷贝

### 浅拷贝

只复制一层，嵌套对象仍是引用。

```python
import copy

original = [1, 2, [3, 4]]

# 浅拷贝方法
shallow1 = original.copy()
shallow2 = list(original)
shallow3 = original[:]
shallow4 = copy.copy(original)

# 修改嵌套对象
shallow1[2][0] = 100
print(original)  # [1, 2, [100, 4]]  ← 也被修改了！
```

### 深拷贝

递归复制所有层级。

```python
import copy

original = [1, 2, [3, 4]]

# 深拷贝
deep = copy.deepcopy(original)

# 修改嵌套对象
deep[2][0] = 100
print(original)  # [1, 2, [3, 4]]  ← 不受影响
print(deep)      # [1, 2, [100, 4]]
```

### 何时用深拷贝

```python
# 嵌套列表
matrix = [[1, 2], [3, 4]]
matrix_copy = copy.deepcopy(matrix)

# 嵌套字典
config = {"db": {"host": "localhost", "port": 3306}}
config_copy = copy.deepcopy(config)

# 包含自定义对象
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

tree = Node(1, [Node(2), Node(3)])
tree_copy = copy.deepcopy(tree)
```

---

## ⚠️ 常见陷阱

### 1. 可变默认参数

```python
# ❌ 危险！
def add_item(item, lst=[]):
    lst.append(item)
    return lst

add_item(1)  # [1]
add_item(2)  # [1, 2]（共享同一个列表！）
add_item(3)  # [1, 2, 3]

# ✅ 正确做法
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### 2. 遍历时修改列表

```python
# ❌ 危险！
lst = [1, 2, 3, 4, 5]
for x in lst:
    if x % 2 == 0:
        lst.remove(x)
print(lst)  # [1, 3, 5]? 可能不正确！

# ✅ 遍历副本
for x in lst[:]:
    if x % 2 == 0:
        lst.remove(x)

# ✅ 使用推导式
lst = [x for x in lst if x % 2 != 0]
```

### 3. 字典遍历时修改

```python
# ❌ 危险！
d = {"a": 1, "b": 2, "c": 3}
for k in d:
    if d[k] > 1:
        del d[k]  # RuntimeError!

# ✅ 遍历副本
for k in list(d.keys()):
    if d[k] > 1:
        del d[k]

# ✅ 使用推导式
d = {k: v for k, v in d.items() if v <= 1}
```

### 4. 引用共享

```python
# ❌ 危险！
matrix = [[0] * 3] * 3
matrix[0][0] = 1
print(matrix)
# [[1, 0, 0], [1, 0, 0], [1, 0, 0]]  ← 三行都变了！

# ✅ 正确做法
matrix = [[0] * 3 for _ in range(3)]
matrix[0][0] = 1
print(matrix)
# [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
```

---

## 📊 JS 对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| 浅拷贝数组 | `lst.copy()` 或 `lst[:]` | `[...arr]` 或 `arr.slice()` |
| 深拷贝 | `copy.deepcopy()` | `JSON.parse(JSON.stringify())` 或 `structuredClone()` |
| 可变默认参数 | 需要避免 | 同样需要避免 |

---

## ✅ 本节要点

1. **不可变**：int, str, tuple, frozenset
2. **可变**：list, dict, set
3. 可变对象作为参数时会被函数修改
4. 浅拷贝只复制一层
5. 嵌套结构需要深拷贝
6. 避免可变默认参数
7. 遍历时不要修改被遍历对象

