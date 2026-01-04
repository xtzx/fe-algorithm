# 04. 字符串

## 🎯 本节目标

- 掌握 Python 字符串操作
- 熟练使用 f-string
- 理解切片语法
- 对比 JS 字符串方法

---

## 📝 字符串创建

### 引号方式

```python
# 单引号和双引号等价
single = 'Hello'
double = "World"

# 三引号：多行字符串
multi = """
这是
多行
字符串
"""

# 三引号也可以用单引号
multi2 = '''
另一种
多行字符串
'''

# 原始字符串（不转义）
raw = r"C:\Users\name"  # 反斜杠不转义
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `'hello'` / `"hello"` | `'hello'` / `"hello"` |
| `"""多行"""` | `` `多行` `` |
| `r"原始"` | `String.raw\`原始\`` |

---

## 🎨 字符串格式化

### 1. f-string（推荐，Python 3.6+）

```python
name = "Alice"
age = 25

# 基本用法
greeting = f"Hello, {name}!"
info = f"Name: {name}, Age: {age}"

# 表达式
result = f"2 + 2 = {2 + 2}"
upper = f"Name: {name.upper()}"

# 格式化数字
pi = 3.14159
formatted = f"Pi: {pi:.2f}"       # "Pi: 3.14"
percent = f"Rate: {0.756:.1%}"    # "Rate: 75.6%"
padded = f"ID: {42:05d}"          # "ID: 00042"

# 对齐
left = f"|{name:<10}|"   # "|Alice     |"
right = f"|{name:>10}|"  # "|     Alice|"
center = f"|{name:^10}|" # "|  Alice   |"
```

### 2. format() 方法

```python
# 位置参数
"{} + {} = {}".format(1, 2, 3)  # "1 + 2 = 3"

# 命名参数
"{name} is {age}".format(name="Bob", age=30)

# 索引参数
"{0} vs {1}".format("Python", "JS")
```

### 3. % 格式化（老式）

```python
"Hello, %s!" % "World"
"Pi is %.2f" % 3.14159
"%d + %d = %d" % (1, 2, 3)
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `f"Hello {name}"` | `` `Hello ${name}` `` |
| `"{} {}".format(a, b)` | 无直接对应 |
| `"%.2f" % num` | `num.toFixed(2)` |

---

## ✂️ 索引与切片

### 索引

```python
s = "Hello World"

# 正向索引（从 0 开始）
s[0]   # 'H'
s[1]   # 'e'
s[4]   # 'o'

# 负向索引（从 -1 开始）
s[-1]  # 'd'（最后一个）
s[-2]  # 'l'
s[-5]  # 'W'
```

### 切片语法

```python
s = "Hello World"

# 基本切片：s[start:end]（包含 start，不包含 end）
s[0:5]    # 'Hello'
s[6:11]   # 'World'

# 省略边界
s[:5]     # 'Hello'（从头开始）
s[6:]     # 'World'（到末尾）
s[:]      # 'Hello World'（完整复制）

# 步长：s[start:end:step]
s[::2]    # 'HloWrd'（每隔一个）
s[1::2]   # 'el ol'（从索引 1 开始，每隔一个）

# 负步长（反转）
s[::-1]   # 'dlroW olleH'
s[-1:-6:-1]  # 'dlroW'
```

### 切片示例

```python
# 获取文件扩展名
filename = "document.pdf"
ext = filename[-4:]  # '.pdf'

# 去掉扩展名
name = filename[:-4]  # 'document'

# 反转字符串
reversed_s = "Python"[::-1]  # 'nohtyP'
```

### JS 对照

| Python | JavaScript | 说明 |
|--------|------------|------|
| `s[0]` | `s[0]` 或 `s.charAt(0)` | 索引 |
| `s[-1]` | `s.at(-1)` 或 `s[s.length-1]` | 负索引 |
| `s[1:4]` | `s.slice(1, 4)` | 切片 |
| `s[::-1]` | `s.split('').reverse().join('')` | 反转 |

---

## 🛠️ 常用方法

### 大小写

```python
s = "Hello World"

s.upper()      # "HELLO WORLD"
s.lower()      # "hello world"
s.capitalize() # "Hello world"（首字母大写）
s.title()      # "Hello World"（每个单词首字母大写）
s.swapcase()   # "hELLO wORLD"
```

### 查找与替换

```python
s = "Hello World"

# 查找
s.find("World")     # 6（找到返回索引）
s.find("Python")    # -1（未找到返回 -1）
s.index("World")    # 6（未找到抛出 ValueError）
s.count("l")        # 3（出现次数）

# 替换
s.replace("World", "Python")  # "Hello Python"
s.replace("l", "L", 1)        # "HeLlo World"（只替换 1 次）
```

### 分割与连接

```python
# 分割
"a,b,c".split(",")        # ['a', 'b', 'c']
"a b c".split()           # ['a', 'b', 'c']（按空白分割）
"a\nb\nc".splitlines()    # ['a', 'b', 'c']

# 连接
",".join(["a", "b", "c"]) # "a,b,c"
" ".join(["Hello", "World"])  # "Hello World"
```

### 去除空白

```python
s = "  Hello World  "

s.strip()   # "Hello World"（两端）
s.lstrip()  # "Hello World  "（左端）
s.rstrip()  # "  Hello World"（右端）

# 指定字符
"###hello###".strip("#")  # "hello"
```

### 判断方法

```python
s = "Hello World"

s.startswith("Hello")  # True
s.endswith("World")    # True
"123".isdigit()        # True
"abc".isalpha()        # True
"abc123".isalnum()     # True
"   ".isspace()        # True
"hello".islower()      # True
"HELLO".isupper()      # True
```

### JS 方法对照表

| Python | JavaScript |
|--------|------------|
| `s.upper()` | `s.toUpperCase()` |
| `s.lower()` | `s.toLowerCase()` |
| `s.strip()` | `s.trim()` |
| `s.split(",")` | `s.split(",")` |
| `",".join(arr)` | `arr.join(",")` |
| `s.replace(a, b)` | `s.replace(a, b)` / `s.replaceAll(a, b)` |
| `s.find(x)` | `s.indexOf(x)` |
| `s.startswith(x)` | `s.startsWith(x)` |
| `s.endswith(x)` | `s.endsWith(x)` |

---

## 🔤 编码

```python
# 字符串 → 字节
s = "你好"
b = s.encode("utf-8")  # b'\xe4\xbd\xa0\xe5\xa5\xbd'

# 字节 → 字符串
s2 = b.decode("utf-8")  # "你好"

# 指定编码
b_gbk = s.encode("gbk")
s_gbk = b_gbk.decode("gbk")
```

---

## 🧩 字符串拼接

```python
# 方式 1：+ 运算符
s = "Hello" + " " + "World"

# 方式 2：f-string（推荐）
s = f"Hello {name}"

# 方式 3：join（多个字符串，性能最好）
parts = ["Hello", "World"]
s = " ".join(parts)

# 方式 4：相邻字符串自动拼接
s = "Hello " "World"  # "Hello World"

# ⚠️ 避免在循环中用 + 拼接（性能差）
# 差
result = ""
for s in strings:
    result += s  # 每次创建新字符串

# 好
result = "".join(strings)
```

---

## ⚠️ 字符串不可变

```python
s = "Hello"

# ❌ 不能修改
s[0] = "h"  # TypeError

# ✅ 只能创建新字符串
s = "h" + s[1:]  # "hello"
s = s.replace("H", "h")  # "hello"
```

---

## ✅ 本节要点

1. 三种引号：单引号、双引号、三引号
2. f-string 是最推荐的格式化方式
3. 切片语法 `[start:end:step]`
4. 负索引 `-1` 表示最后一个
5. `[::-1]` 可以反转字符串
6. 字符串是不可变的
7. `join()` 比 `+` 拼接性能更好

