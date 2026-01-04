# 01. pathlib - 现代文件路径

## 🎯 本节目标

- 掌握 Path 对象的创建和操作
- 熟练使用路径拼接和属性
- 了解文件操作方法

---

## 📝 Path 对象创建

```python
from pathlib import Path

# 当前目录
p = Path(".")
p = Path.cwd()

# 主目录
p = Path.home()

# 指定路径
p = Path("/usr/local/bin")
p = Path("data/file.txt")

# 从字符串
p = Path("data") / "subdir" / "file.txt"
```

### JS 对照

```javascript
// Node.js path 模块
const path = require('path');
const p = path.join('data', 'subdir', 'file.txt');
```

---

## 🔗 路径拼接

使用 `/` 运算符拼接路径（Python 特色）。

```python
from pathlib import Path

# / 运算符拼接
base = Path("/home/user")
full = base / "documents" / "file.txt"
print(full)  # /home/user/documents/file.txt

# 也可以用 joinpath
full = base.joinpath("documents", "file.txt")
```

### ⚠️ 不要用字符串拼接

```python
# ❌ 错误
p = "/home/user" + "/documents"

# ✅ 正确
p = Path("/home/user") / "documents"
```

---

## 📋 路径属性

```python
from pathlib import Path

p = Path("/home/user/documents/report.txt")

print(p.name)      # report.txt（文件名）
print(p.stem)      # report（不含扩展名）
print(p.suffix)    # .txt（扩展名）
print(p.parent)    # /home/user/documents
print(p.parts)     # ('/', 'home', 'user', 'documents', 'report.txt')

# 多级扩展名
p = Path("archive.tar.gz")
print(p.suffixes)  # ['.tar', '.gz']

# 绝对路径
p = Path("relative/path")
print(p.absolute())
print(p.resolve())  # 解析符号链接
```

---

## 🔍 路径检查

```python
from pathlib import Path

p = Path("somefile.txt")

# 存在性检查
p.exists()      # 是否存在
p.is_file()     # 是否是文件
p.is_dir()      # 是否是目录
p.is_symlink()  # 是否是符号链接
p.is_absolute() # 是否是绝对路径
```

---

## 📁 目录操作

```python
from pathlib import Path

# 创建目录
p = Path("new_dir")
p.mkdir()                          # 创建
p.mkdir(exist_ok=True)             # 已存在不报错
p.mkdir(parents=True, exist_ok=True)  # 创建父目录

# 删除目录（必须为空）
p.rmdir()

# 遍历目录
for item in Path(".").iterdir():
    print(item)
```

---

## 🔎 文件查找

### glob - 模式匹配

```python
from pathlib import Path

p = Path(".")

# 匹配当前目录
for f in p.glob("*.py"):
    print(f)

# 递归匹配（rglob）
for f in p.rglob("*.py"):
    print(f)

# 复杂模式
for f in p.glob("**/*.txt"):  # 同 rglob("*.txt")
    print(f)

for f in p.glob("data[0-9].csv"):
    print(f)
```

---

## 📄 文件操作

```python
from pathlib import Path

p = Path("example.txt")

# 读取
content = p.read_text()           # 读取文本
content = p.read_text(encoding="utf-8")
data = p.read_bytes()             # 读取二进制

# 写入
p.write_text("Hello, World!")
p.write_text("你好", encoding="utf-8")
p.write_bytes(b"binary data")

# 删除文件
p.unlink()
p.unlink(missing_ok=True)  # 不存在不报错

# 重命名/移动
new_p = p.rename("new_name.txt")
new_p = p.replace("target.txt")  # 覆盖目标
```

---

## 🔄 路径转换

```python
from pathlib import Path

p = Path("data/file.txt")

# 转字符串
str(p)  # "data/file.txt"

# 改变扩展名
p.with_suffix(".md")  # data/file.md

# 改变文件名
p.with_name("other.txt")  # data/other.txt

# 改变 stem
p.with_stem("new")  # data/new.txt (Python 3.9+)
```

---

## 🆚 pathlib vs os.path

| 操作 | pathlib | os.path |
|------|---------|---------|
| 拼接 | `p / "sub"` | `os.path.join(p, "sub")` |
| 存在 | `p.exists()` | `os.path.exists(p)` |
| 是文件 | `p.is_file()` | `os.path.isfile(p)` |
| 文件名 | `p.name` | `os.path.basename(p)` |
| 目录名 | `p.parent` | `os.path.dirname(p)` |
| 绝对路径 | `p.resolve()` | `os.path.abspath(p)` |
| 读取 | `p.read_text()` | `open(p).read()` |

**推荐使用 pathlib**：
- 面向对象，更直观
- 链式操作
- 跨平台
- Python 3.4+ 标准库

---

## 🎯 实际应用

### 查找所有 Python 文件

```python
from pathlib import Path

def find_python_files(directory):
    return list(Path(directory).rglob("*.py"))

files = find_python_files(".")
print(f"找到 {len(files)} 个 Python 文件")
```

### 批量重命名

```python
from pathlib import Path

def rename_files(directory, pattern, new_suffix):
    for f in Path(directory).glob(pattern):
        new_name = f.with_suffix(new_suffix)
        f.rename(new_name)
        print(f"重命名: {f} -> {new_name}")
```

---

## ✅ 本节要点

1. `Path` 对象代替字符串路径
2. `/` 运算符拼接路径
3. `name`, `stem`, `suffix`, `parent` 获取路径部分
4. `exists()`, `is_file()`, `is_dir()` 检查路径
5. `read_text()`, `write_text()` 读写文件
6. `glob()`, `rglob()` 查找文件


