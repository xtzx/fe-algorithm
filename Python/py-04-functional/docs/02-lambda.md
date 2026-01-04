# 02. lambda 表达式

## 🎯 本节目标

- 掌握 lambda 语法
- 理解使用场景和限制
- 对比 JS 箭头函数

---

## 📝 lambda 语法

`lambda` 用于创建匿名函数（没有名字的函数）。

### 基本语法

```python
# lambda 参数: 表达式
square = lambda x: x**2
print(square(5))  # 25

# 等价于
def square(x):
    return x**2
```

### 多参数

```python
# 两个参数
add = lambda x, y: x + y
print(add(3, 5))  # 8

# 多个参数
multiply = lambda a, b, c: a * b * c
print(multiply(2, 3, 4))  # 24
```

### 无参数

```python
get_answer = lambda: 42
print(get_answer())  # 42
```

### 默认参数

```python
power = lambda x, n=2: x**n
print(power(5))    # 25 (默认 n=2)
print(power(5, 3)) # 125
```

---

## 🎯 使用场景

### 1. 作为排序的 key

```python
# 按长度排序
words = ["apple", "pie", "banana"]
sorted(words, key=lambda w: len(w))
# ['pie', 'apple', 'banana']

# 按第二个元素排序
pairs = [(1, 3), (2, 1), (3, 2)]
sorted(pairs, key=lambda p: p[1])
# [(2, 1), (3, 2), (1, 3)]
```

### 2. 与 map/filter 配合

```python
numbers = [1, 2, 3, 4, 5]

# map
squares = list(map(lambda x: x**2, numbers))
# [1, 4, 9, 16, 25]

# filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4]
```

### 3. 作为回调函数

```python
# 事件处理
def on_click(handler):
    # 模拟点击事件
    handler("button clicked")

on_click(lambda event: print(f"Event: {event}"))
# Event: button clicked
```

### 4. 字典排序

```python
users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
]

sorted(users, key=lambda u: u["age"])
```

---

## ⚠️ 限制

### 只能包含表达式

```python
# ✅ 正确：表达式
square = lambda x: x**2

# ❌ 错误：不能包含语句
# lambda x: print(x)  # 可以，但 print 返回 None
# lambda x: if x > 0: x else -x  # 语法错误

# ✅ 使用条件表达式
abs_val = lambda x: x if x > 0 else -x
```

### 不能包含赋值

```python
# ❌ 错误：不能赋值
# lambda x: y = x + 1

# ✅ 正确：使用普通函数
def add_one(x):
    y = x + 1
    return y
```

### 不能包含 return

```python
# lambda 自动返回表达式结果
square = lambda x: x**2  # 自动返回 x**2

# ❌ 错误：不能显式 return
# lambda x: return x**2
```

---

## 🆚 lambda vs 普通函数

| 特性 | lambda | def 函数 |
|------|--------|---------|
| 语法 | `lambda x: x**2` | `def f(x): return x**2` |
| 名字 | 匿名 | 有名字 |
| 复杂度 | 简单表达式 | 可包含多条语句 |
| 文档字符串 | 不支持 | 支持 |
| 使用场景 | 简单回调 | 复杂逻辑 |

### 何时用 lambda

✅ **适合**：
- 简单的一行表达式
- 作为参数传递（如 key、回调）
- 临时使用，不需要名字

❌ **不适合**：
- 复杂逻辑（用普通函数）
- 需要文档字符串
- 需要调试（lambda 没有名字）

---

## 🔄 JS 箭头函数对照

### 基本语法

```python
# Python lambda
square = lambda x: x**2
add = lambda x, y: x + y
```

```javascript
// JavaScript 箭头函数
const square = x => x**2;
const add = (x, y) => x + y;
```

### 多行

```python
# Python：lambda 只能单行
# 需要多行用普通函数
def complex_func(x):
    y = x * 2
    z = y + 1
    return z
```

```javascript
// JavaScript：箭头函数可以多行
const complexFunc = x => {
    const y = x * 2;
    const z = y + 1;
    return z;
};
```

### this 绑定

```python
# Python：没有 this 概念
class MyClass:
    def method(self):
        return lambda x: x + self.value
```

```javascript
// JavaScript：箭头函数继承外层 this
class MyClass {
    method() {
        return x => x + this.value;  // this 绑定到 MyClass
    }
}
```

---

## 🎭 常见用法示例

### 条件表达式

```python
# 返回较大值
max_val = lambda a, b: a if a > b else b

# 判断奇偶
is_even = lambda x: x % 2 == 0

# 绝对值
abs_val = lambda x: x if x >= 0 else -x
```

### 嵌套 lambda

```python
# 返回函数的函数
make_adder = lambda n: lambda x: x + n

add_5 = make_adder(5)
print(add_5(10))  # 15
```

### 列表操作

```python
# 提取字段
users = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
names = list(map(lambda u: u["name"], users))
# ['Alice', 'Bob']

# 过滤
adults = list(filter(lambda u: u["age"] >= 18, users))
```

---

## ⚠️ 常见陷阱

### 1. 循环中的 lambda

```python
# ❌ 问题：所有 lambda 捕获最后一个值
funcs = []
for i in range(3):
    funcs.append(lambda x: x + i)

print(funcs[0](10))  # 12（所有都是 i=2）
print(funcs[1](10))  # 12
print(funcs[2](10))  # 12

# ✅ 解决：使用默认参数
funcs = []
for i in range(3):
    funcs.append(lambda x, i=i: x + i)

print(funcs[0](10))  # 10
print(funcs[1](10))  # 11
print(funcs[2](10))  # 12
```

### 2. 过度使用 lambda

```python
# ❌ 可读性差
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# ✅ 更清晰
evens = [x for x in range(10) if x % 2 == 0]
result = [x**2 for x in evens]
```

---

## ✅ 本节要点

1. `lambda` 创建匿名函数：`lambda 参数: 表达式`
2. 只能包含表达式，不能有语句
3. 适合简单回调，不适合复杂逻辑
4. 循环中使用默认参数避免变量绑定问题
5. JS 箭头函数功能更强大（多行、this 绑定）

