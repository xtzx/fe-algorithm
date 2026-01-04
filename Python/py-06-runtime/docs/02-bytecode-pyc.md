# 02. 字节码与 .pyc 文件

## 本节目标

- 深入理解字节码结构
- 了解 .pyc 文件的生成和失效
- 掌握 `__pycache__` 机制

---

## 字节码详解

### Code Object 结构

```python
def example(a, b):
    """示例函数"""
    x = a + b
    return x * 2

code = example.__code__

print(f"co_name: {code.co_name}")           # 函数名
print(f"co_filename: {code.co_filename}")   # 源文件
print(f"co_firstlineno: {code.co_firstlineno}")  # 起始行号
print(f"co_argcount: {code.co_argcount}")   # 参数数量
print(f"co_varnames: {code.co_varnames}")   # 局部变量名
print(f"co_consts: {code.co_consts}")       # 常量表
print(f"co_code: {code.co_code}")           # 字节码
```

### 重要的 Code Object 属性

| 属性 | 说明 |
|------|------|
| `co_name` | 函数/代码块名称 |
| `co_argcount` | 位置参数数量 |
| `co_varnames` | 局部变量名元组 |
| `co_consts` | 常量元组 |
| `co_names` | 全局变量名元组 |
| `co_code` | 字节码字节串 |
| `co_stacksize` | 栈大小 |
| `co_flags` | 编译标志 |

---

## dis 模块详解

### 基本反汇编

```python
import dis

def calculate(x, y):
    if x > y:
        return x - y
    else:
        return x + y

dis.dis(calculate)
```

输出：
```
  2           0 LOAD_FAST                0 (x)
              2 LOAD_FAST                1 (y)
              4 COMPARE_OP               4 (>)
              6 POP_JUMP_IF_FALSE       16

  3           8 LOAD_FAST                0 (x)
             10 LOAD_FAST                1 (y)
             12 BINARY_SUBTRACT
             14 RETURN_VALUE

  5     >>   16 LOAD_FAST                0 (x)
             18 LOAD_FAST                1 (y)
             20 BINARY_ADD
             22 RETURN_VALUE
```

### 字节码指令解读

```
  3           8 LOAD_FAST                0 (x)
  │           │ │                        │  └── 参数说明
  │           │ │                        └── 操作数
  │           │ └── 指令名
  │           └── 字节偏移量
  └── 源代码行号
```

### 常用字节码指令

```python
import dis

# 查看所有操作码
print(dis.opmap)

# 常用指令分类：
# 加载：LOAD_FAST, LOAD_CONST, LOAD_GLOBAL, LOAD_ATTR
# 存储：STORE_FAST, STORE_NAME, STORE_ATTR
# 运算：BINARY_ADD, BINARY_SUBTRACT, BINARY_MULTIPLY
# 比较：COMPARE_OP
# 跳转：JUMP_FORWARD, POP_JUMP_IF_FALSE, JUMP_ABSOLUTE
# 函数：CALL_FUNCTION, RETURN_VALUE
```

---

## .pyc 文件

### 什么是 .pyc

.pyc 是编译后的字节码缓存文件，避免每次导入都重新编译。

### 生成位置

```
project/
├── module.py
└── __pycache__/
    └── module.cpython-312.pyc
         │          │      └── 扩展名
         │          └── Python 版本
         └── 模块名
```

### 手动编译

```python
import py_compile
import compileall

# 编译单个文件
py_compile.compile('module.py')

# 编译整个目录
compileall.compile_dir('src/', force=True)
```

### 查看 .pyc 内容

```python
import marshal
import dis
import struct

def read_pyc(filename):
    with open(filename, 'rb') as f:
        # 魔数 (4 bytes) - Python 版本标识
        magic = f.read(4)
        print(f"Magic: {magic.hex()}")

        # PEP 552: 额外字段 (4 bytes)
        f.read(4)

        # 时间戳 (4 bytes)
        timestamp = struct.unpack('I', f.read(4))[0]
        print(f"Timestamp: {timestamp}")

        # 源文件大小 (4 bytes)
        size = struct.unpack('I', f.read(4))[0]
        print(f"Source size: {size}")

        # 代码对象
        code = marshal.load(f)
        return code

# code = read_pyc('__pycache__/module.cpython-312.pyc')
# dis.dis(code)
```

---

## .pyc 何时重新生成

### 自动重新生成条件

1. **源文件时间戳变化**: 修改了 .py 文件
2. **Python 版本变化**: Magic Number 不匹配
3. **.pyc 不存在**: 首次导入

### 强制重新编译

```bash
# 删除所有 .pyc
find . -type d -name __pycache__ -exec rm -rf {} +

# 或使用 Python
python3 -B script.py  # -B 禁止生成 .pyc

# 强制重新编译
python3 -m compileall -f .
```

### 环境变量

```bash
# 禁止生成 .pyc
export PYTHONDONTWRITEBYTECODE=1

# 指定 .pyc 目录
export PYTHONPYCACHEPREFIX=/tmp/pycache
```

---

## __pycache__ 机制

### 目录结构

```
mypackage/
├── __init__.py
├── module1.py
├── module2.py
└── __pycache__/
    ├── __init__.cpython-312.pyc
    ├── module1.cpython-312.pyc
    └── module2.cpython-312.pyc
```

### 多版本共存

```
__pycache__/
├── module.cpython-310.pyc
├── module.cpython-311.pyc
└── module.cpython-312.pyc
```

### 为什么这样设计

1. **整洁**: 缓存文件统一存放
2. **多版本**: 不同 Python 版本互不干扰
3. **便于清理**: 删除一个目录即可

---

## 优化级别

```bash
# 正常编译
python3 -m py_compile module.py

# 优化级别 1 (-O): 移除 assert，生成 .opt-1.pyc
python3 -O -m py_compile module.py

# 优化级别 2 (-OO): 额外移除 docstring
python3 -OO -m py_compile module.py
```

```
__pycache__/
├── module.cpython-312.pyc
├── module.cpython-312.opt-1.pyc
└── module.cpython-312.opt-2.pyc
```

---

## 字节码版本兼容性

### Magic Number

每个 Python 版本有唯一的 Magic Number：

```python
import importlib.util
print(importlib.util.MAGIC_NUMBER.hex())
# Python 3.12: a70d0d0a
```

### 不兼容情况

- Python 3.10 的 .pyc **不能**在 3.12 使用
- 同版本号的 .pyc 可以跨平台使用

---

## 实际应用

### 部署时预编译

```bash
# 预编译所有文件
python3 -m compileall -b .  # -b 生成在源文件旁边

# Docker 中减少启动时间
python3 -m compileall /app
```

### 只分发 .pyc

```python
# 简单的代码保护（不安全，可反编译）
# 分发 __pycache__ 中的 .pyc 文件
```

### 调试缓存问题

```python
# 查看模块来源
import mymodule
print(mymodule.__file__)
print(mymodule.__cached__)
```

---

## 本节要点

1. **Code Object**: 包含字节码和元信息的对象
2. **dis 模块**: 反汇编查看字节码指令
3. **.pyc 文件**: 字节码缓存，避免重复编译
4. **__pycache__**: 统一存放，支持多版本
5. **重新生成**: 源文件变化或 Python 版本变化
6. **优化级别**: -O 和 -OO 生成优化的 .pyc

