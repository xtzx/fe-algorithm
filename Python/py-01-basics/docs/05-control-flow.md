# 05. 控制流

## 🎯 本节目标

- 掌握条件语句
- 掌握循环语句
- 理解 Python 循环的 else 子句
- 学会 match-case（Python 3.10+）

---

## 🔀 条件语句

### if / elif / else

```python
age = 18

if age < 13:
    print("儿童")
elif age < 18:
    print("青少年")
else:
    print("成年人")
```

### JS 对照

```javascript
// Python: if / elif / else
// JS:     if / else if / else

if (age < 13) {
    console.log("儿童");
} else if (age < 18) {
    console.log("青少年");
} else {
    console.log("成年人");
}
```

### 单行条件

```python
# 三元表达式
status = "成年" if age >= 18 else "未成年"

# 单行 if
if age >= 18: print("成年")
```

### 多条件判断

```python
# and / or
if age >= 18 and has_id:
    print("可以进入")

if is_vip or has_ticket:
    print("欢迎入场")

# 链式比较
if 0 < age < 120:
    print("有效年龄")
```

---

## 🔄 for 循环

### 基本语法

```python
# 遍历列表
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Hello":
    print(char)

# 遍历字典
data = {"a": 1, "b": 2}
for key in data:
    print(key, data[key])

for key, value in data.items():
    print(f"{key}: {value}")
```

### range() 函数

```python
# range(stop)：0 到 stop-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, stop)
for i in range(1, 6):
    print(i)  # 1, 2, 3, 4, 5

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# 倒序
for i in range(5, 0, -1):
    print(i)  # 5, 4, 3, 2, 1
```

### enumerate()：带索引遍历

```python
fruits = ["apple", "banana", "cherry"]

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# 指定起始索引
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")  # 1, 2, 3
```

### zip()：并行遍历

```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age}")
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `for x in list:` | `for (const x of list)` |
| `range(5)` | `[...Array(5).keys()]` |
| `enumerate(list)` | `list.entries()` 或 `list.forEach((x, i))` |
| `zip(a, b)` | `a.map((x, i) => [x, b[i]])` |

---

## 🔁 while 循环

```python
count = 0
while count < 5:
    print(count)
    count += 1

# 无限循环
while True:
    user_input = input("输入 'quit' 退出: ")
    if user_input == "quit":
        break
```

---

## ⏹️ break, continue, else

### break：跳出循环

```python
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4
```

### continue：跳过当前迭代

```python
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4
```

### 循环的 else 子句（Python 特有）

```python
# else 在循环正常结束时执行（未被 break 中断）
for i in range(5):
    if i == 10:
        break
else:
    print("循环正常结束")  # 会执行

# 被 break 中断时不执行 else
for i in range(5):
    if i == 3:
        break
else:
    print("不会执行")
```

**实际应用：查找**

```python
# 查找元素
target = 7
for num in [1, 3, 5, 7, 9]:
    if num == target:
        print(f"找到 {target}")
        break
else:
    print(f"未找到 {target}")
```

---

## 🎯 match-case（Python 3.10+）

类似 JS 的 switch，但更强大。

### 基本语法

```python
command = "start"

match command:
    case "start":
        print("启动中...")
    case "stop":
        print("停止中...")
    case "restart":
        print("重启中...")
    case _:  # 默认情况（类似 default）
        print("未知命令")
```

### 模式匹配

```python
# 匹配值
match value:
    case 0:
        print("零")
    case 1 | 2 | 3:  # 多个值
        print("1, 2 或 3")
    case _:
        print("其他")

# 匹配序列
match point:
    case (0, 0):
        print("原点")
    case (x, 0):
        print(f"X 轴上，x = {x}")
    case (0, y):
        print(f"Y 轴上，y = {y}")
    case (x, y):
        print(f"点 ({x}, {y})")

# 匹配字典
match data:
    case {"type": "user", "name": name}:
        print(f"用户：{name}")
    case {"type": "admin"}:
        print("管理员")

# 带条件（guard）
match value:
    case x if x > 0:
        print("正数")
    case x if x < 0:
        print("负数")
    case _:
        print("零")
```

### JS 对照

```javascript
// JS switch
switch (command) {
    case "start":
        console.log("启动中...");
        break;
    case "stop":
        console.log("停止中...");
        break;
    default:
        console.log("未知命令");
}
```

| 特性 | Python match | JS switch |
|------|-------------|-----------|
| 默认情况 | `case _` | `default` |
| 贯穿 | 无（更安全） | 需要 `break` |
| 模式匹配 | ✅ 支持 | ❌ 不支持 |
| 解构 | ✅ 支持 | ❌ 不支持 |

---

## 🧩 推导式（Comprehension）

### 列表推导式

```python
# 传统方式
squares = []
for x in range(10):
    squares.append(x ** 2)

# 列表推导式
squares = [x ** 2 for x in range(10)]

# 带条件
evens = [x for x in range(10) if x % 2 == 0]

# 嵌套
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6]
```

### 字典推导式

```python
# {key: value for item in iterable}
squares = {x: x ** 2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 集合推导式

```python
# {value for item in iterable}
unique_lengths = {len(word) for word in ["apple", "banana", "cherry"]}
# {5, 6}
```

### 生成器表达式

```python
# (value for item in iterable)
gen = (x ** 2 for x in range(10))  # 不立即计算，惰性求值
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `[x*2 for x in arr]` | `arr.map(x => x*2)` |
| `[x for x in arr if x > 0]` | `arr.filter(x => x > 0)` |
| `{k: v for k, v in items}` | `Object.fromEntries(items)` |

---

## ✅ 本节要点

1. `elif` 而不是 `else if`
2. 缩进决定代码块（没有大括号）
3. `for x in iterable` 类似 JS `for...of`
4. `range()` 生成数字序列
5. 循环的 `else` 在正常结束时执行
6. `match-case` 支持模式匹配（Python 3.10+）
7. 推导式是 Python 的强大语法糖

