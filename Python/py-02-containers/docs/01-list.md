# 01. 列表 list

## 🎯 本节目标

- 掌握列表的创建与操作
- 熟练使用切片
- 理解排序与复制

---

## 📝 创建列表

```python
# 字面量
lst = [1, 2, 3, 4, 5]

# list() 构造函数
lst = list()              # 空列表
lst = list("hello")       # ['h', 'e', 'l', 'l', 'o']
lst = list(range(5))      # [0, 1, 2, 3, 4]

# 列表推导式
lst = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# 重复
lst = [0] * 5             # [0, 0, 0, 0, 0]
```

### JS 对照

```javascript
// JS 创建数组
const arr = [1, 2, 3];
const arr = Array(5).fill(0);
const arr = Array.from({length: 5}, (_, i) => i);
const arr = [...Array(5).keys()];
```

---

## 🔧 基本操作

### 添加元素

```python
lst = [1, 2, 3]

# append：末尾添加单个元素
lst.append(4)         # [1, 2, 3, 4]

# extend：扩展多个元素
lst.extend([5, 6])    # [1, 2, 3, 4, 5, 6]

# insert：指定位置插入
lst.insert(0, 0)      # [0, 1, 2, 3, 4, 5, 6]
lst.insert(-1, 5.5)   # 在倒数第一个前插入

# + 运算符：创建新列表
new_lst = lst + [7, 8]
```

### 删除元素

```python
lst = [1, 2, 3, 4, 5]

# pop：弹出并返回
last = lst.pop()      # 5，lst = [1, 2, 3, 4]
first = lst.pop(0)    # 1，lst = [2, 3, 4]

# remove：删除第一个匹配的值
lst.remove(3)         # lst = [2, 4]

# del：删除指定索引
del lst[0]            # lst = [4]

# clear：清空
lst.clear()           # lst = []
```

### 查找

```python
lst = [1, 2, 3, 2, 4]

# in：检查存在
2 in lst              # True
5 in lst              # False

# index：获取索引（不存在会报错）
lst.index(2)          # 1（第一个匹配）
lst.index(2, 2)       # 3（从索引 2 开始找）

# count：计数
lst.count(2)          # 2
```

### JS 对照表

| Python | JavaScript |
|--------|------------|
| `lst.append(x)` | `arr.push(x)` |
| `lst.extend([x, y])` | `arr.push(x, y)` |
| `lst.insert(0, x)` | `arr.unshift(x)` |
| `lst.pop()` | `arr.pop()` |
| `lst.pop(0)` | `arr.shift()` |
| `lst.remove(x)` | `arr.splice(arr.indexOf(x), 1)` |
| `x in lst` | `arr.includes(x)` |
| `lst.index(x)` | `arr.indexOf(x)` |
| `lst.count(x)` | `arr.filter(i => i === x).length` |

---

## ✂️ 切片

切片是 Python 最强大的特性之一。

### 基本语法

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# lst[start:end] - 不包含 end
lst[2:5]      # [2, 3, 4]
lst[:5]       # [0, 1, 2, 3, 4]（从头开始）
lst[5:]       # [5, 6, 7, 8, 9]（到末尾）
lst[:]        # 完整复制

# 负索引
lst[-3:]      # [7, 8, 9]（最后 3 个）
lst[:-3]      # [0, 1, 2, 3, 4, 5, 6]（除了最后 3 个）
```

### 步长

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# lst[start:end:step]
lst[::2]      # [0, 2, 4, 6, 8]（每隔一个）
lst[1::2]     # [1, 3, 5, 7, 9]（奇数索引）
lst[::3]      # [0, 3, 6, 9]（每隔两个）

# 负步长（反向）
lst[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]（反转）
lst[::-2]     # [9, 7, 5, 3, 1]（反向每隔一个）
lst[7:2:-1]   # [7, 6, 5, 4, 3]
```

### 切片赋值

```python
lst = [0, 1, 2, 3, 4]

# 替换部分元素
lst[1:3] = [10, 20, 30]  # [0, 10, 20, 30, 3, 4]

# 删除部分元素
lst[1:4] = []            # [0, 3, 4]

# 插入
lst[1:1] = [1, 2]        # [0, 1, 2, 3, 4]
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `lst[1:4]` | `arr.slice(1, 4)` |
| `lst[::-1]` | `arr.slice().reverse()` |
| `lst[::2]` | `arr.filter((_, i) => i % 2 === 0)` |

---

## 📊 排序

### 原地排序：sort()

```python
lst = [3, 1, 4, 1, 5, 9, 2, 6]

# 默认升序
lst.sort()            # [1, 1, 2, 3, 4, 5, 6, 9]

# 降序
lst.sort(reverse=True)  # [9, 6, 5, 4, 3, 2, 1, 1]

# 自定义排序键
words = ["apple", "pie", "banana"]
words.sort(key=len)   # ["pie", "apple", "banana"]

# 复杂排序
users = [{"name": "Bob", "age": 30}, {"name": "Alice", "age": 25}]
users.sort(key=lambda u: u["age"])
```

### 返回新列表：sorted()

```python
lst = [3, 1, 4, 1, 5]

# 不修改原列表
new_lst = sorted(lst)           # [1, 1, 3, 4, 5]
new_lst = sorted(lst, reverse=True)

# 可用于任何可迭代对象
sorted("hello")       # ['e', 'h', 'l', 'l', 'o']
sorted({3, 1, 2})     # [1, 2, 3]
```

### 反转

```python
lst = [1, 2, 3]

# 原地反转
lst.reverse()         # [3, 2, 1]

# 返回新列表
new_lst = lst[::-1]
new_lst = list(reversed(lst))
```

---

## 📋 复制

### 浅拷贝

```python
original = [1, 2, [3, 4]]

# 方式 1：切片
copy1 = original[:]

# 方式 2：copy()
copy2 = original.copy()

# 方式 3：list()
copy3 = list(original)

# ⚠️ 浅拷贝：嵌套对象仍是引用
copy1[2][0] = 100
print(original)  # [1, 2, [100, 4]]  ← 也被修改了！
```

### 深拷贝

```python
import copy

original = [1, 2, [3, 4]]
deep = copy.deepcopy(original)

deep[2][0] = 100
print(original)  # [1, 2, [3, 4]]  ← 不受影响
```

---

## 🎭 列表作为栈和队列

### 栈（LIFO）

```python
stack = []
stack.append(1)   # 入栈
stack.append(2)
stack.pop()       # 出栈 → 2
```

### 队列（FIFO）

**⚠️ 重要**：不要用 `list.pop(0)` 作队列，效率低 O(n)

```python
from collections import deque

# 创建双端队列
queue = deque()
queue.append(1)      # 右端入队
queue.append(2)
queue.popleft()      # 左端出队 → 1

# 也可以从左边操作
queue.appendleft(0)  # 左端入队
queue.pop()          # 右端出队 → 2
```

**deque 的优势**：
- `append` / `popleft` 都是 O(1)
- `list.pop(0)` 是 O(n)，因为需要移动所有元素
- 支持双端操作

**deque 常用方法**：
```python
d = deque([1, 2, 3])
d.append(4)          # 右端添加
d.appendleft(0)      # 左端添加
d.pop()              # 右端弹出
d.popleft()          # 左端弹出
d.extend([5, 6])     # 右端扩展
d.extendleft([-1])   # 左端扩展（注意顺序）
len(d)               # 长度
d[0]                 # 索引访问
```

---

## ✅ 本节要点

1. `append` 添加单个，`extend` 添加多个
2. 切片 `[start:end:step]`，负数表示倒数
3. `[::-1]` 反转列表
4. `sort()` 原地排序，`sorted()` 返回新列表
5. 浅拷贝 vs 深拷贝：嵌套对象的区别
6. 队列用 `deque`，不要用 `pop(0)`

