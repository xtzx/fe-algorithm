# 05. 推导式 Comprehensions

## 🎯 本节目标

- 掌握各类推导式
- 理解生成器表达式
- 写出 Pythonic 代码

---

## 📝 列表推导式

### 基本语法

```python
# [表达式 for 变量 in 可迭代对象]
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 等价的循环写法
squares = []
for x in range(10):
    squares.append(x**2)
```

### 条件过滤

```python
# [表达式 for 变量 in 可迭代对象 if 条件]
evens = [x for x in range(20) if x % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# 复杂条件
filtered = [x for x in range(100) if x % 2 == 0 and x % 3 == 0]
# [0, 6, 12, 18, 24, 30, ...]
```

### 条件表达式

```python
# [真值 if 条件 else 假值 for 变量 in 可迭代对象]
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
# ['even', 'odd', 'even', 'odd', 'even']

# 注意位置：
# if 在 for 后面 → 过滤
# if else 在 for 前面 → 条件表达式
```

### 嵌套推导式

```python
# 嵌套循环
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 展平
flat = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 等价于
flat = []
for row in matrix:
    for num in row:
        flat.append(num)

# 生成嵌套列表
grid = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```

### 多变量

```python
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# 带条件
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
# [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
```

---

## 📖 字典推导式

```python
# {键表达式: 值表达式 for 变量 in 可迭代对象}
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 从两个列表创建
keys = ["a", "b", "c"]
values = [1, 2, 3]
d = {k: v for k, v in zip(keys, values)}
# {"a": 1, "b": 2, "c": 3}

# 条件过滤
d = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# 键值互换
original = {"a": 1, "b": 2, "c": 3}
reversed_d = {v: k for k, v in original.items()}
# {1: "a", 2: "b", 3: "c"}

# 转换
words = ["hello", "world"]
lengths = {word: len(word) for word in words}
# {"hello": 5, "world": 5}
```

---

## 🔵 集合推导式

```python
# {表达式 for 变量 in 可迭代对象}
squares = {x**2 for x in range(-5, 6)}
# {0, 1, 4, 9, 16, 25}（自动去重）

# 条件过滤
evens = {x for x in range(20) if x % 2 == 0}

# 从字符串
chars = {c.lower() for c in "Hello World"}
# {'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'}
```

---

## ⚡ 生成器表达式

```python
# (表达式 for 变量 in 可迭代对象)
gen = (x**2 for x in range(10))
print(gen)  # <generator object ...>

# 惰性求值：不立即计算
# 遍历时才计算
for val in gen:
    print(val)

# 转为列表
lst = list(x**2 for x in range(10))

# 用于函数参数可省略括号
sum(x**2 for x in range(10))
max(len(word) for word in words)
```

### 生成器 vs 列表推导式

```python
# 列表推导式：立即计算，占用内存
lst = [x**2 for x in range(1000000)]  # 立即创建 100 万个元素

# 生成器表达式：惰性计算，省内存
gen = (x**2 for x in range(1000000))  # 只创建生成器对象
```

| 特性 | 列表推导式 | 生成器表达式 |
|------|-----------|-------------|
| 语法 | `[...]` | `(...)` |
| 求值 | 立即 | 惰性 |
| 内存 | 全部存储 | 按需生成 |
| 重复遍历 | 可以 | 只能一次 |
| 适用场景 | 需要多次访问 | 一次遍历/大数据 |

---

## 🆚 推导式 vs map/filter

### map

```python
# map 方式
squares = list(map(lambda x: x**2, range(10)))

# 推导式方式（更 Pythonic）
squares = [x**2 for x in range(10)]
```

### filter

```python
# filter 方式
evens = list(filter(lambda x: x % 2 == 0, range(20)))

# 推导式方式
evens = [x for x in range(20) if x % 2 == 0]
```

### 组合

```python
# map + filter
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# 推导式（更清晰）
result = [x**2 for x in range(10) if x % 2 == 0]
```

> **Python 风格指南**：推导式比 map/filter 更 Pythonic

---

## 📊 实际应用

### 数据转换

```python
# 提取字段
users = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
names = [u["name"] for u in users]

# 格式化
formatted = [f"User: {u['name']}" for u in users]
```

### 数据过滤

```python
# 过滤有效数据
valid_users = [u for u in users if u["age"] >= 18]

# 过滤空值
clean_data = [x for x in data if x is not None]
clean_data = [x for x in data if x]  # 过滤 Falsy
```

### 嵌套结构处理

```python
# 展平嵌套列表
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]

# 提取嵌套字段
data = [{"items": [1, 2]}, {"items": [3, 4]}]
all_items = [item for d in data for item in d["items"]]
```

### 创建查找表

```python
# ID 到名称的映射
id_to_name = {u["id"]: u["name"] for u in users}

# 按条件分组
by_category = {cat: [x for x in items if x["cat"] == cat]
               for cat in categories}
```

---

## ⚠️ 注意事项

### 不要过度嵌套

```python
# ❌ 太复杂，难以阅读
result = [[y * 2 for y in x if y > 0] for x in matrix if sum(x) > 10]

# ✅ 拆分或用循环
result = []
for x in matrix:
    if sum(x) > 10:
        result.append([y * 2 for y in x if y > 0])
```

### 副作用

```python
# ❌ 不要用推导式执行副作用
[print(x) for x in items]  # 创建了无用的 None 列表

# ✅ 用普通循环
for x in items:
    print(x)
```

---

## ✅ 本节要点

1. 列表推导式：`[expr for x in iterable if condition]`
2. 字典推导式：`{k: v for k, v in items}`
3. 集合推导式：`{expr for x in iterable}`
4. 生成器表达式：`(expr for x in iterable)` 惰性求值
5. 推导式比 map/filter 更 Pythonic
6. 大数据用生成器表达式节省内存
7. 避免过度嵌套，保持可读性

