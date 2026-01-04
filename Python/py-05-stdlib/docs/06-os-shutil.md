# 06. os 与 shutil - 系统操作

## 本节目标

- 掌握环境变量操作
- 学会文件和目录操作
- 了解进程执行

---

## os.environ - 环境变量

```python
import os

# 获取环境变量
home = os.environ.get("HOME")
path = os.environ.get("PATH")

# 使用默认值
debug = os.environ.get("DEBUG", "false")

# 直接访问（可能抛出 KeyError）
try:
    api_key = os.environ["API_KEY"]
except KeyError:
    api_key = None

# getenv 方法
api_key = os.getenv("API_KEY", "default")
```

### 设置环境变量

```python
import os

# 设置（仅当前进程）
os.environ["MY_VAR"] = "value"

# 删除
del os.environ["MY_VAR"]
```

---

## os 基本操作

```python
import os

# 当前目录
print(os.getcwd())

# 改变目录
os.chdir("/tmp")

# 列出目录
print(os.listdir("."))

# 创建目录
os.mkdir("new_dir")
os.makedirs("path/to/dir", exist_ok=True)

# 删除
os.remove("file.txt")     # 删除文件
os.rmdir("empty_dir")     # 删除空目录

# 重命名
os.rename("old.txt", "new.txt")

# 文件信息
stat = os.stat("file.txt")
print(stat.st_size)   # 大小
print(stat.st_mtime)  # 修改时间
```

### os.path（老式路径操作）

```python
import os.path

# 拼接
path = os.path.join("dir", "subdir", "file.txt")

# 路径部分
print(os.path.dirname("/path/to/file.txt"))   # /path/to
print(os.path.basename("/path/to/file.txt"))  # file.txt
print(os.path.splitext("file.txt"))           # ('file', '.txt')

# 检查
print(os.path.exists("file.txt"))
print(os.path.isfile("file.txt"))
print(os.path.isdir("directory"))

# 绝对路径
print(os.path.abspath("relative/path"))
```

**建议**：优先使用 `pathlib`，更现代更直观。

---

## shutil - 高级文件操作

```python
import shutil

# 复制文件
shutil.copy("src.txt", "dst.txt")           # 复制内容
shutil.copy2("src.txt", "dst.txt")          # 复制内容+元数据

# 复制目录
shutil.copytree("src_dir", "dst_dir")
shutil.copytree("src", "dst", dirs_exist_ok=True)  # Python 3.8+

# 移动
shutil.move("src.txt", "dst.txt")
shutil.move("src_dir", "dst_dir")

# 删除目录（包括内容）
shutil.rmtree("directory")
shutil.rmtree("directory", ignore_errors=True)

# 磁盘使用
usage = shutil.disk_usage("/")
print(f"总计: {usage.total / 1e9:.1f} GB")
print(f"已用: {usage.used / 1e9:.1f} GB")
print(f"可用: {usage.free / 1e9:.1f} GB")
```

### 归档操作

```python
import shutil

# 创建归档
shutil.make_archive("backup", "zip", "source_dir")
shutil.make_archive("backup", "tar", "source_dir")
shutil.make_archive("backup", "gztar", "source_dir")

# 解压
shutil.unpack_archive("backup.zip", "extract_dir")
```

---

## tempfile - 临时文件

```python
import tempfile

# 临时文件
with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
    f.write("临时内容")
    print(f.name)  # 临时文件路径

# 临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    print(tmpdir)  # 临时目录路径
    # 使用完自动删除

# 获取临时目录路径
print(tempfile.gettempdir())
```

---

## subprocess - 进程执行

```python
import subprocess

# 简单执行
result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
print(result.stdout)
print(result.returncode)

# 执行命令字符串
result = subprocess.run("ls -la | head", shell=True, capture_output=True, text=True)

# 检查返回值
try:
    subprocess.run(["false"], check=True)
except subprocess.CalledProcessError as e:
    print(f"命令失败，返回码: {e.returncode}")
```

### 与 os.system 对比

```python
import os
import subprocess

# os.system（简单但功能有限）
exit_code = os.system("ls -la")

# subprocess（推荐，功能强大）
result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
```

---

## 实际应用

### 安全删除目录

```python
import shutil
from pathlib import Path

def safe_rmtree(path):
    """安全删除目录，先确认存在"""
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        print(f"已删除: {path}")
    else:
        print(f"目录不存在: {path}")
```

### 复制带过滤

```python
import shutil

def ignore_patterns(src, names):
    """忽略 .git 和 __pycache__"""
    return [n for n in names if n in (".git", "__pycache__")]

shutil.copytree("src", "dst", ignore=ignore_patterns)

# 或使用内置函数
shutil.copytree("src", "dst", ignore=shutil.ignore_patterns("*.pyc", ".git"))
```

### 批量移动文件

```python
import shutil
from pathlib import Path

def organize_by_extension(src_dir, dst_dir):
    """按扩展名整理文件"""
    src = Path(src_dir)
    dst = Path(dst_dir)

    for file in src.iterdir():
        if file.is_file():
            ext = file.suffix.lower() or "no_extension"
            target_dir = dst / ext.lstrip(".")
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target_dir / file.name))
```

---

## 本节要点

1. `os.environ` 读写环境变量
2. `os.path` 路径操作（推荐用 pathlib）
3. `shutil.copy/move/rmtree` 文件目录操作
4. `tempfile` 创建临时文件和目录
5. `subprocess.run` 执行外部命令


