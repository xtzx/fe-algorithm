# 07. 文件 I/O

## 🎯 本节目标

- 掌握 print() 的高级用法
- 学会 input() 获取用户输入
- 熟练使用文件读写
- 理解 with 语句

---

## 🖨️ print() 函数

### 基本用法

```python
print("Hello, World!")
print("Hello", "World")  # Hello World（空格分隔）
```

### 参数详解

```python
# sep：分隔符（默认空格）
print("a", "b", "c", sep=",")  # a,b,c
print("a", "b", "c", sep=" | ")  # a | b | c

# end：结尾字符（默认换行）
print("Hello", end=" ")
print("World")  # Hello World（同一行）

# file：输出目标
with open("output.txt", "w") as f:
    print("写入文件", file=f)

# flush：立即刷新缓冲区
import time
for i in range(5):
    print(f"\r进度：{i+1}/5", end="", flush=True)
    time.sleep(1)
```

### 格式化输出

```python
name = "Alice"
age = 25

# f-string
print(f"Name: {name}, Age: {age}")

# format
print("Name: {}, Age: {}".format(name, age))

# 对齐
print(f"|{name:10}|")   # |Alice     |（左对齐，宽度 10）
print(f"|{name:>10}|")  # |     Alice|（右对齐）
print(f"|{name:^10}|")  # |  Alice   |（居中）
```

---

## ⌨️ input() 函数

```python
# 基本用法
name = input("请输入你的名字: ")
print(f"你好, {name}!")

# ⚠️ input() 总是返回字符串
age_str = input("请输入你的年龄: ")
age = int(age_str)  # 需要手动转换

# 简洁写法
age = int(input("请输入你的年龄: "))

# 处理多个输入
data = input("输入多个数字（空格分隔）: ")
numbers = [int(x) for x in data.split()]
```

---

## 📂 文件操作

### 打开文件

```python
# open(file, mode, encoding)
f = open("data.txt", "r", encoding="utf-8")
content = f.read()
f.close()  # 必须关闭！
```

### 文件模式

| 模式 | 说明 |
|------|------|
| `"r"` | 只读（默认） |
| `"w"` | 写入（覆盖） |
| `"a"` | 追加 |
| `"x"` | 创建（文件存在则报错） |
| `"b"` | 二进制模式 |
| `"t"` | 文本模式（默认） |
| `"+"` | 读写模式 |

```python
# 常用组合
"r"   # 只读文本
"rb"  # 只读二进制
"w"   # 写入文本（覆盖）
"wb"  # 写入二进制
"a"   # 追加文本
"r+"  # 读写文本
```

---

## 📖 读取文件

### read()：读取全部

```python
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()  # 整个文件内容
```

### readline()：读取一行

```python
with open("data.txt", "r") as f:
    line = f.readline()  # 第一行
    line2 = f.readline()  # 第二行
```

### readlines()：读取所有行

```python
with open("data.txt", "r") as f:
    lines = f.readlines()  # 列表，每行一个元素
    # 注意：每行末尾有 \n
```

### 逐行迭代（推荐）

```python
with open("data.txt", "r") as f:
    for line in f:  # 内存高效
        print(line.strip())  # strip() 去掉换行符
```

---

## ✏️ 写入文件

### write()

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("第二行\n")
```

### writelines()

```python
lines = ["第一行\n", "第二行\n", "第三行\n"]

with open("output.txt", "w") as f:
    f.writelines(lines)  # 不会自动加换行符
```

### 追加模式

```python
with open("log.txt", "a") as f:
    f.write("新的日志条目\n")
```

---

## 🔒 with 语句

### 为什么用 with？

```python
# ❌ 传统方式：容易忘记关闭
f = open("data.txt")
try:
    content = f.read()
finally:
    f.close()

# ✅ with 语句：自动关闭
with open("data.txt") as f:
    content = f.read()
# 离开 with 块自动调用 f.close()
```

### 同时打开多个文件

```python
with open("input.txt", "r") as f_in, \
     open("output.txt", "w") as f_out:
    f_out.write(f_in.read())

# Python 3.10+ 可以用括号
with (
    open("input.txt", "r") as f_in,
    open("output.txt", "w") as f_out
):
    f_out.write(f_in.read())
```

---

## 🗂️ 常用文件操作

### pathlib（推荐）

```python
from pathlib import Path

# 创建路径对象
p = Path("data/file.txt")

# 读取
content = p.read_text(encoding="utf-8")

# 写入
p.write_text("Hello", encoding="utf-8")

# 路径操作
p.exists()       # 是否存在
p.is_file()      # 是否是文件
p.is_dir()       # 是否是目录
p.name           # 文件名
p.stem           # 文件名（无扩展名）
p.suffix         # 扩展名
p.parent         # 父目录

# 路径拼接
new_path = Path("data") / "subdir" / "file.txt"
```

### os 模块

```python
import os

# 路径操作
os.path.exists("file.txt")
os.path.isfile("file.txt")
os.path.isdir("data")
os.path.join("data", "file.txt")

# 目录操作
os.makedirs("path/to/dir", exist_ok=True)
os.listdir(".")
os.remove("file.txt")
os.rmdir("empty_dir")

# 当前目录
os.getcwd()
os.chdir("/path/to/dir")
```

---

## 🔄 JSON 文件

```python
import json

# 写入 JSON
data = {"name": "Alice", "age": 25}
with open("data.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# 读取 JSON
with open("data.json", "r") as f:
    data = json.load(f)

# 字符串转换
json_str = json.dumps(data)
data = json.loads(json_str)
```

---

## 📊 CSV 文件

```python
import csv

# 写入 CSV
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])
    writer.writerow(["Alice", 25])
    writer.writerow(["Bob", 30])

# 读取 CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# 字典方式
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])
```

---

## ⚠️ 常见问题

### 编码问题

```python
# 指定编码（推荐）
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Windows 默认可能是 GBK
with open("file.txt", "r", encoding="gbk") as f:
    content = f.read()
```

### 文件不存在

```python
from pathlib import Path

path = Path("file.txt")
if path.exists():
    content = path.read_text()
else:
    print("文件不存在")
```

### 大文件处理

```python
# ❌ 一次性读取（内存可能不够）
with open("huge.txt") as f:
    content = f.read()

# ✅ 逐行处理
with open("huge.txt") as f:
    for line in f:
        process(line)
```

---

## ✅ 本节要点

1. `print()` 的 `sep`、`end`、`file`、`flush` 参数
2. `input()` 总是返回字符串
3. 文件模式：`r`(读)、`w`(写)、`a`(追加)、`b`(二进制)
4. 必须使用 `with` 语句（自动关闭）
5. 逐行读取大文件更高效
6. 推荐使用 `pathlib` 处理路径
7. 注意指定 `encoding="utf-8"`

