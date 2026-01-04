# 🔧 15 - Python 调试技巧

> 前端视角：从 console.log 到 Python 调试大师

---

## 目录

1. [Print 调试（快速版）](#1-print-调试快速版)
2. [断点调试](#2-断点调试)
3. [常见错误类型与排查](#3-常见错误类型与排查)
4. [异常处理最佳实践](#4-异常处理最佳实践)
5. [性能分析](#5-性能分析)
6. [调试技巧汇总](#6-调试技巧汇总)

---

## 前端 vs Python 调试工具对照

| 功能 | JavaScript | Python |
|------|------------|--------|
| **快速打印** | `console.log()` | `print()` |
| **格式化打印** | `console.table()` | `pprint()` |
| **断点调试** | Chrome DevTools | pdb / VS Code |
| **性能分析** | Performance API | cProfile / timeit |
| **日志记录** | console.warn/error | logging 模块 |
| **错误追踪** | Error.stack | traceback 模块 |

---

## 1. Print 调试（快速版）

### 1.1 基础 print

```python
# 最简单的调试方式（相当于 console.log）
x = 42
print(x)  # 42

# 打印多个变量
name = "Alice"
age = 25
print(name, age)  # Alice 25

# 打印变量名和值（Python 3.8+）
print(f"{x=}")      # x=42
print(f"{name=}")   # name='Alice'
print(f"{age=}, {name=}")  # age=25, name='Alice'

# 这个语法非常适合调试！
data = [1, 2, 3]
print(f"{data=}, {len(data)=}")  # data=[1, 2, 3], len(data)=3
```

### 1.2 f-string 格式化

```python
# f-string 是 Python 3.6+ 的特性（类似 JS 模板字符串）
name = "Alice"
score = 95.5678

# 基础用法
print(f"Name: {name}, Score: {score}")

# 格式化数字
print(f"Score: {score:.2f}")      # Score: 95.57（保留2位小数）
print(f"Score: {score:>10.2f}")   # Score:      95.57（右对齐，宽度10）
print(f"Percentage: {score:.1%}") # Percentage: 9556.8%

# 日期格式化
from datetime import datetime
now = datetime.now()
print(f"Time: {now:%Y-%m-%d %H:%M:%S}")  # Time: 2024-01-15 14:30:45

# 调试模式（= 语法）
x = 10
y = 20
print(f"{x + y = }")  # x + y = 30
print(f"{x * y = }")  # x * y = 200
```

### 1.3 pprint 美化输出

```python
from pprint import pprint

# 普通 print 对复杂数据结构不友好
data = {
    "users": [
        {"name": "Alice", "age": 25, "skills": ["Python", "ML"]},
        {"name": "Bob", "age": 30, "skills": ["Java", "Go", "Docker"]},
    ],
    "metadata": {"version": "1.0", "count": 2}
}

# 普通 print（一行，难以阅读）
print(data)

# pprint（格式化，易于阅读）
pprint(data)
# 输出:
# {'metadata': {'count': 2, 'version': '1.0'},
#  'users': [{'age': 25, 'name': 'Alice', 'skills': ['Python', 'ML']},
#            {'age': 30,
#             'name': 'Bob',
#             'skills': ['Java', 'Go', 'Docker']}]}

# 控制宽度和深度
pprint(data, width=60, depth=2)
```

### 1.4 logging 模块

```python
import logging

# 配置日志（放在文件开头）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 使用不同级别的日志
logging.debug("调试信息：变量 x = 10")      # 开发时使用
logging.info("一般信息：用户登录成功")      # 记录正常流程
logging.warning("警告：磁盘空间不足")       # 需要注意的问题
logging.error("错误：数据库连接失败")       # 错误，但程序可继续
logging.critical("严重：系统崩溃")          # 致命错误

# 输出示例:
# 2024-01-15 14:30:45,123 - DEBUG - 调试信息：变量 x = 10
# 2024-01-15 14:30:45,124 - INFO - 一般信息：用户登录成功
```

**日志级别对比**:

| 级别 | 数值 | 用途 | JS 对应 |
|------|:----:|------|---------|
| DEBUG | 10 | 开发调试 | console.debug |
| INFO | 20 | 一般信息 | console.info |
| WARNING | 30 | 警告 | console.warn |
| ERROR | 40 | 错误 | console.error |
| CRITICAL | 50 | 严重错误 | - |

```python
# 实际项目中的日志配置
import logging

def setup_logging():
    # 创建 logger
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)

    # 格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# 使用
logger = setup_logging()
logger.info("程序启动")
logger.debug("调试信息（只写入文件）")
```

---

## 2. 断点调试

### 2.1 pdb 基础

```python
# pdb 是 Python 内置的调试器（类似 Chrome DevTools 的 Sources 面板）

# 方法 1：在代码中插入断点
import pdb

def calculate(a, b):
    result = a + b
    pdb.set_trace()  # 程序会在这里暂停
    result = result * 2
    return result

calculate(3, 5)

# 方法 2：Python 3.7+ 更简洁的方式
def calculate(a, b):
    result = a + b
    breakpoint()  # 等同于 pdb.set_trace()
    result = result * 2
    return result
```

**pdb 常用命令**：

| 命令 | 简写 | 功能 | 类比 Chrome DevTools |
|------|------|------|---------------------|
| `help` | `h` | 显示帮助 | - |
| `list` | `l` | 显示当前代码 | Sources 面板 |
| `next` | `n` | 执行下一行（不进入函数） | Step Over (F10) |
| `step` | `s` | 执行下一行（进入函数） | Step Into (F11) |
| `continue` | `c` | 继续执行到下一个断点 | Resume (F8) |
| `print expr` | `p expr` | 打印表达式的值 | Console |
| `pp expr` | - | 美化打印 | - |
| `where` | `w` | 显示调用栈 | Call Stack |
| `up` | `u` | 跳到上一层调用栈 | - |
| `down` | `d` | 跳到下一层调用栈 | - |
| `quit` | `q` | 退出调试 | - |

```python
# pdb 实战示例
def process_data(data):
    result = []
    for item in data:
        breakpoint()  # 在循环中设置断点
        processed = item * 2
        result.append(processed)
    return result

# 运行后，在 pdb 提示符下:
# (Pdb) p item        # 打印当前 item
# (Pdb) p result      # 打印当前 result
# (Pdb) n             # 执行下一行
# (Pdb) c             # 继续执行（到下一次循环的断点）
```

### 2.2 VS Code 断点调试

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 当前文件",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: 带参数",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "args": ["--input", "data.csv", "--output", "result.json"],
            "console": "integratedTerminal"
        }
    ]
}
```

**VS Code 调试快捷键**：

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| 开始调试 | F5 | 启动调试 |
| 停止调试 | Shift+F5 | 停止 |
| 重启调试 | Ctrl+Shift+F5 | 重启 |
| 单步跳过 | F10 | Step Over |
| 单步进入 | F11 | Step Into |
| 单步跳出 | Shift+F11 | Step Out |
| 继续 | F5 | Continue |
| 切换断点 | F9 | Toggle Breakpoint |

### 2.3 Jupyter 中的调试

```python
# 方法 1：使用 %debug 魔法命令
def buggy_function(x):
    return 10 / x

# 执行出错后
buggy_function(0)  # ZeroDivisionError

# 然后在下一个 cell 运行
%debug
# 这会打开交互式调试器，可以检查错误发生时的状态

# 方法 2：在代码中设置断点
from IPython.core.debugger import set_trace

def process(data):
    for i, item in enumerate(data):
        if i == 2:
            set_trace()  # Jupyter 友好的断点
        print(item)

# 方法 3：使用 %%debug cell magic
%%debug
x = 1
y = 0
z = x / y

# 方法 4：启用自动调试（出错时自动进入调试器）
%pdb on
# 之后任何错误都会自动进入调试器

# 关闭自动调试
%pdb off
```

---

## 3. 常见错误类型与排查

### 3.1 TypeError

```python
# TypeError: 类型不匹配

# 错误示例 1：不能将字符串和整数相加
# result = "age: " + 25  # ❌ TypeError

# 修复
result = "age: " + str(25)  # ✅
result = f"age: {25}"       # ✅ 更好的方式

# 错误示例 2：不可调用的对象
x = 10
# x()  # ❌ TypeError: 'int' object is not callable

# 错误示例 3：参数数量错误
def greet(name, age):
    print(f"Hello {name}, you are {age}")

# greet("Alice")  # ❌ TypeError: missing 1 required positional argument

# 修复
greet("Alice", 25)  # ✅

# 🔍 前端对比
# JS 中 "age: " + 25 会自动转换为 "age: 25"
# Python 更严格，需要显式转换
```

### 3.2 ValueError

```python
# ValueError: 值不合法

# 错误示例 1：转换失败
# int("abc")  # ❌ ValueError: invalid literal for int()

# 修复：先检查或用 try-except
def safe_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default

print(safe_int("123"))   # 123
print(safe_int("abc"))   # 0

# 错误示例 2：解包数量不匹配
# a, b = [1, 2, 3]  # ❌ ValueError: too many values to unpack

# 修复
a, b, c = [1, 2, 3]      # ✅
a, b, *rest = [1, 2, 3]  # ✅ a=1, b=2, rest=[3]
```

### 3.3 KeyError / IndexError

```python
# KeyError: 字典键不存在
user = {"name": "Alice", "age": 25}

# user["email"]  # ❌ KeyError: 'email'

# 修复方法
email = user.get("email")           # ✅ 返回 None
email = user.get("email", "N/A")    # ✅ 返回默认值
if "email" in user:                  # ✅ 先检查
    email = user["email"]

# IndexError: 索引越界
arr = [1, 2, 3]

# arr[10]  # ❌ IndexError: list index out of range

# 修复方法
if len(arr) > 10:
    print(arr[10])

# 或使用 try-except
try:
    print(arr[10])
except IndexError:
    print("索引越界")

# 🔍 前端对比
# JS 中 arr[10] 返回 undefined，不会报错
# Python 更严格，会抛出 IndexError
```

### 3.4 AttributeError

```python
# AttributeError: 对象没有该属性

name = "Alice"
# name.append("!")  # ❌ AttributeError: 'str' object has no attribute 'append'

# 调试技巧：查看对象有哪些属性
print(dir(name))  # 列出所有属性和方法
print(type(name)) # 查看类型

# 使用 hasattr 检查
if hasattr(name, 'append'):
    name.append("!")
else:
    print("字符串没有 append 方法")

# 常见场景：None 对象
result = None
# result.split()  # ❌ AttributeError: 'NoneType' object has no attribute 'split'

# 修复
if result is not None:
    result.split()

# 或使用短路求值
result and result.split()
```

### 3.5 ImportError / ModuleNotFoundError

```python
# ModuleNotFoundError: 模块未安装

# import some_package  # ❌ ModuleNotFoundError

# 排查步骤：
# 1. 检查是否安装
#    pip list | grep some_package
#    pip show some_package

# 2. 检查虚拟环境
#    which python
#    pip list

# 3. 检查拼写
#    import sklearn  # 不是 scikit-learn

# ImportError: 导入路径问题
# from mypackage import mymodule  # 可能路径不对

# 排查
import sys
print(sys.path)  # 查看 Python 搜索路径

# 添加自定义路径
sys.path.append('/path/to/your/module')
```

### 3.6 维度不匹配（NumPy/PyTorch）

```python
import numpy as np

# 这是 AI 开发中最常见的错误之一！

# 错误示例：矩阵乘法维度不匹配
A = np.array([[1, 2], [3, 4]])      # 形状: (2, 2)
B = np.array([[1, 2, 3], [4, 5, 6]]) # 形状: (2, 3)
C = np.array([1, 2, 3])              # 形状: (3,)

# A @ C  # ❌ ValueError: shapes (2,2) and (3,) not aligned

# 排查技巧：打印形状
print(f"A.shape = {A.shape}")  # (2, 2)
print(f"B.shape = {B.shape}")  # (2, 3)
print(f"C.shape = {C.shape}")  # (3,)

# 矩阵乘法规则：(m, n) @ (n, p) = (m, p)
result = A @ B  # ✅ (2, 2) @ (2, 3) = (2, 3)

# 修复维度问题的常用方法
C_2d = C.reshape(3, 1)    # 变成列向量 (3, 1)
C_2d = C[:, np.newaxis]   # 另一种方式
C_2d = np.expand_dims(C, axis=1)  # 又一种方式

# 广播错误
# A + C  # 形状不兼容

# 广播规则：从右往左对齐，维度要么相同，要么其中一个是 1
```

---

## 4. 异常处理最佳实践

### 4.1 try-except 细粒度捕获

```python
# ❌ 不好的做法：捕获所有异常
try:
    result = do_something()
except:  # 太宽泛
    pass

# ❌ 也不好：捕获 Exception
try:
    result = do_something()
except Exception:  # 还是太宽泛
    pass

# ✅ 好的做法：捕获具体异常
try:
    result = int(user_input)
except ValueError:
    print("请输入有效的数字")

# ✅ 捕获多个具体异常
try:
    data = fetch_data()
    result = process(data)
except ConnectionError:
    print("网络连接失败")
except TimeoutError:
    print("请求超时")
except ValueError as e:
    print(f"数据格式错误: {e}")

# ✅ 获取异常信息
import traceback

try:
    risky_operation()
except SomeError as e:
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print(f"完整堆栈:\n{traceback.format_exc()}")
```

### 4.2 try-except-else-finally

```python
# 完整的异常处理结构
try:
    # 可能出错的代码
    file = open("data.txt", "r")
    data = file.read()
except FileNotFoundError:
    # 处理特定错误
    print("文件不存在")
    data = None
except PermissionError:
    # 处理另一种错误
    print("没有读取权限")
    data = None
else:
    # 没有异常时执行（可选）
    print(f"成功读取 {len(data)} 字节")
finally:
    # 无论如何都会执行（清理资源）
    if 'file' in locals() and not file.closed:
        file.close()
        print("文件已关闭")
```

### 4.3 自定义异常

```python
# 定义自定义异常
class ValidationError(Exception):
    """数据验证错误"""
    pass

class AuthenticationError(Exception):
    """认证错误"""
    def __init__(self, message, user_id=None):
        super().__init__(message)
        self.user_id = user_id

# 使用自定义异常
def validate_age(age):
    if not isinstance(age, int):
        raise ValidationError(f"年龄必须是整数，收到: {type(age).__name__}")
    if age < 0 or age > 150:
        raise ValidationError(f"年龄必须在 0-150 之间，收到: {age}")
    return True

def login(user_id, password):
    if not check_password(user_id, password):
        raise AuthenticationError("密码错误", user_id=user_id)

# 捕获
try:
    validate_age("twenty")
except ValidationError as e:
    print(f"验证失败: {e}")

try:
    login("alice", "wrong_password")
except AuthenticationError as e:
    print(f"登录失败: {e}, 用户: {e.user_id}")
```

### 4.4 上下文管理器（with 语句）

```python
# 上下文管理器自动处理资源释放（类似 JS 的 try-finally）

# ✅ 文件操作
with open("data.txt", "r") as f:
    data = f.read()
# 文件自动关闭，即使出错也会关闭

# ✅ 数据库连接
import sqlite3

with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
# 连接自动关闭

# ✅ 锁
import threading

lock = threading.Lock()
with lock:
    # 临界区代码
    pass
# 锁自动释放

# 自定义上下文管理器
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self.start
        print(f"耗时: {self.elapsed:.4f} 秒")
        return False  # 不抑制异常

# 使用
with Timer():
    # 要计时的代码
    sum(range(1000000))
# 输出: 耗时: 0.0234 秒

# 使用 contextlib 简化
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"耗时: {time.time() - start:.4f} 秒")

with timer():
    sum(range(1000000))
```

---

## 5. 性能分析

### 5.1 简单计时

```python
import time

# 方法 1：time.time()
start = time.time()
result = sum(range(1000000))
end = time.time()
print(f"耗时: {end - start:.4f} 秒")

# 方法 2：time.perf_counter()（更精确）
start = time.perf_counter()
result = sum(range(1000000))
end = time.perf_counter()
print(f"耗时: {end - start:.6f} 秒")

# 方法 3：封装成装饰器
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.4f} 秒")
        return result
    return wrapper

@timer
def slow_function():
    return sum(range(10000000))

slow_function()  # slow_function 耗时: 0.2345 秒
```

### 5.2 Jupyter %timeit

```python
# %timeit 自动多次运行取平均，更准确

# 单行计时
%timeit sum(range(1000))
# 输出: 12.3 µs ± 456 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# 多行计时
%%timeit
total = 0
for i in range(1000):
    total += i

# 比较不同实现
%timeit sum(range(1000))           # 使用内置 sum
%timeit sum([i for i in range(1000)])  # 列表推导式

# 控制运行次数
%timeit -n 100 -r 3 sum(range(1000))
# -n: 每次测试的循环次数
# -r: 重复测试的次数
```

### 5.3 cProfile 性能分析

```python
import cProfile
import pstats

def main():
    """要分析的主函数"""
    result = []
    for i in range(1000):
        result.append(expensive_operation(i))
    return result

def expensive_operation(x):
    return sum(range(x))

# 方法 1：直接分析
cProfile.run('main()')

# 方法 2：保存结果并分析
profiler = cProfile.Profile()
profiler.enable()

main()  # 运行要分析的代码

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')  # 按累计时间排序
stats.print_stats(10)  # 打印前 10 个

# 方法 3：命令行运行
# python -m cProfile -s cumulative your_script.py

# 输出解读:
# ncalls: 调用次数
# tottime: 函数本身耗时（不含子函数）
# percall: 每次调用平均耗时
# cumtime: 累计耗时（含子函数）
# filename:lineno(function): 函数位置
```

### 5.4 line_profiler 逐行分析

```bash
# 安装
pip install line_profiler
```

```python
# 使用 @profile 装饰器标记要分析的函数
@profile
def slow_function():
    result = []
    for i in range(1000):
        result.append(i ** 2)  # 这行可能较慢

    total = sum(result)        # 这行也要检查
    return total

# 命令行运行
# kernprof -l -v your_script.py

# 输出示例:
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      3                                           @profile
#      4                                           def slow_function():
#      5         1         10.0     10.0      0.0      result = []
#      6      1001       5000.0      5.0     45.5      for i in range(1000):
#      7      1000       5500.0      5.5     50.0          result.append(i ** 2)
#      8         1        500.0    500.0      4.5      total = sum(result)
#      9         1          0.0      0.0      0.0      return total
```

### 5.5 memory_profiler 内存分析

```bash
# 安装
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def memory_hungry():
    # 创建大列表
    big_list = [i ** 2 for i in range(1000000)]

    # 处理数据
    result = sum(big_list)

    # 删除大列表
    del big_list

    return result

memory_hungry()

# 输出示例:
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      3     50.0 MiB     50.0 MiB   @profile
#      4                             def memory_hungry():
#      5     88.5 MiB     38.5 MiB       big_list = [i ** 2 for i in range(1000000)]
#      6     88.5 MiB      0.0 MiB       result = sum(big_list)
#      7     50.0 MiB    -38.5 MiB       del big_list
#      8     50.0 MiB      0.0 MiB       return result

# Jupyter 中使用
%load_ext memory_profiler
%memit sum(range(1000000))
```

---

## 6. 调试技巧汇总

### 6.1 二分法定位问题

```python
# 当代码很长，不知道哪里出错时，使用二分法

def complex_function(data):
    # 第一部分
    step1_result = process_step1(data)
    print(f"Step 1 完成: {step1_result[:5]}...")  # 检查点 1

    # 第二部分
    step2_result = process_step2(step1_result)
    print(f"Step 2 完成: {step2_result[:5]}...")  # 检查点 2

    # 第三部分
    step3_result = process_step3(step2_result)
    print(f"Step 3 完成")  # 检查点 3

    return step3_result

# 通过检查点输出，定位问题在哪一步
```

### 6.2 最小化复现

```python
# 当遇到复杂问题时，先创建最小复现案例

# ❌ 原始复杂代码
def complex_ml_pipeline(data_path):
    data = load_data(data_path)
    data = preprocess(data)
    features = extract_features(data)
    model = train_model(features)
    # 某处出错...

# ✅ 最小复现
# 1. 确定是哪个函数出错
# 2. 用最简单的输入复现问题

def test_preprocess():
    # 使用简单的测试数据
    simple_data = {"a": [1, 2, None], "b": [4, 5, 6]}
    result = preprocess(simple_data)
    print(result)

test_preprocess()  # 更容易定位问题
```

### 6.3 查看源码（inspect 模块）

```python
import inspect

# 查看函数源码
import pandas as pd
print(inspect.getsource(pd.DataFrame.merge))

# 查看函数签名
print(inspect.signature(pd.DataFrame.merge))

# 查看函数定义位置
print(inspect.getfile(pd.DataFrame.merge))

# 查看对象的所有成员
print(inspect.getmembers(pd.DataFrame))

# 查看调用栈
def outer():
    inner()

def inner():
    # 打印调用栈
    for frame in inspect.stack():
        print(f"{frame.filename}:{frame.lineno} in {frame.function}")

outer()
```

### 6.4 利用 AI 辅助调试

```python
# 当遇到难以理解的错误时，可以这样向 AI 提问：

# 1. 提供完整错误信息
"""
我遇到了这个错误：

Traceback (most recent call last):
  File "main.py", line 10, in <module>
    result = process(data)
  File "main.py", line 5, in process
    return data.groupby('category').sum()
AttributeError: 'list' object has no attribute 'groupby'

我的代码是：
```python
data = [{"category": "A", "value": 1}, ...]
result = process(data)
```

请帮我分析原因和解决方案。
"""

# 2. 提供上下文
"""
- Python 版本：3.10
- Pandas 版本：2.0
- 我想实现的功能是...
- 我已经尝试过...
"""

# 3. 使用 repr() 获取精确的对象表示
data = [1, 2, 3]
print(f"data = {repr(data)}")  # data = [1, 2, 3]
print(f"type = {type(data)}")  # type = <class 'list'>
```

### 6.5 常用调试代码片段

```python
# 放在代码中快速调试

# 1. 快速打印变量信息
def debug_var(name, var):
    print(f"[DEBUG] {name}:")
    print(f"  type: {type(var)}")
    print(f"  value: {repr(var)[:100]}")
    if hasattr(var, 'shape'):
        print(f"  shape: {var.shape}")
    if hasattr(var, '__len__'):
        print(f"  len: {len(var)}")

# 使用
debug_var("data", data)

# 2. 条件断点
def process(items):
    for i, item in enumerate(items):
        if item == "problematic_value":  # 只在特定条件下断点
            breakpoint()
        # 处理逻辑

# 3. 异常后进入调试
import sys

def debug_on_error(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)

sys.excepthook = debug_on_error

# 4. 记录函数调用
def trace_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"→ {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"← {func.__name__} returned {repr(result)[:50]}")
        return result
    return wrapper

@trace_calls
def my_function(x, y):
    return x + y
```

---

## 📚 调试工具速查表

| 场景 | 工具 | 命令/用法 |
|------|------|----------|
| 快速打印 | print | `print(f"{x=}")` |
| 格式化打印 | pprint | `pprint(data)` |
| 断点调试 | pdb | `breakpoint()` |
| VS Code 调试 | debugpy | F5 启动 |
| Jupyter 调试 | %debug | 错误后运行 |
| 函数计时 | timeit | `%timeit func()` |
| 性能分析 | cProfile | `cProfile.run()` |
| 逐行分析 | line_profiler | `@profile` |
| 内存分析 | memory_profiler | `@profile` |

---

## ➡️ 下一步

学完本节后，继续学习 [16-Docker入门.md](./16-Docker入门.md)

