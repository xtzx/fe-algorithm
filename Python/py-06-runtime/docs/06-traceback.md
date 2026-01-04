# 06. 错误与 Traceback

## 本节目标

- 学会读懂 Python 的错误信息
- 理解异常层级结构
- 掌握异常链和自定义异常

---

## 读懂 Traceback

### 基本结构

```python
def func_c():
    return 1 / 0

def func_b():
    return func_c()

def func_a():
    return func_b()

func_a()
```

输出：
```
Traceback (most recent call last):
  File "example.py", line 10, in <module>
    func_a()
  File "example.py", line 8, in func_a
    return func_b()
  File "example.py", line 5, in func_b
    return func_c()
  File "example.py", line 2, in func_c
    return 1 / 0
ZeroDivisionError: division by zero
```

### 阅读顺序

```
从上到下 = 从调用者到被调用者
最后一行 = 异常类型和消息
倒数第二段 = 实际出错的代码
```

### 各部分含义

```
Traceback (most recent call last):  ← 标题
  File "example.py", line 10, in <module>  ← 文件、行号、作用域
    func_a()  ← 该行代码
  ...
ZeroDivisionError: division by zero  ← 异常类型: 消息
```

---

## 异常层级结构

```
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── StopIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── AssertionError
    ├── AttributeError
    ├── BufferError
    ├── EOFError
    ├── ImportError
    │   └── ModuleNotFoundError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── MemoryError
    ├── NameError
    │   └── UnboundLocalError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── TimeoutError
    ├── RuntimeError
    │   └── RecursionError
    ├── SyntaxError
    │   └── IndentationError
    ├── TypeError
    └── ValueError
        └── UnicodeError
```

### 常见异常

| 异常 | 说明 | 示例 |
|------|------|------|
| `TypeError` | 类型错误 | `"a" + 1` |
| `ValueError` | 值错误 | `int("abc")` |
| `KeyError` | 字典键不存在 | `d["missing"]` |
| `IndexError` | 索引越界 | `lst[100]` |
| `AttributeError` | 属性不存在 | `obj.missing` |
| `FileNotFoundError` | 文件不存在 | `open("missing")` |
| `ZeroDivisionError` | 除以零 | `1/0` |

---

## 捕获异常

### 基本语法

```python
try:
    result = risky_operation()
except SomeException as e:
    handle_error(e)
else:
    # 没有异常时执行
    process(result)
finally:
    # 总是执行
    cleanup()
```

### 捕获多个异常

```python
try:
    operation()
except (TypeError, ValueError) as e:
    print(f"类型或值错误: {e}")
except KeyError:
    print("键不存在")
except Exception as e:
    print(f"其他错误: {e}")
```

### 获取异常信息

```python
import traceback

try:
    1 / 0
except Exception as e:
    # 异常类型和消息
    print(f"类型: {type(e).__name__}")
    print(f"消息: {e}")

    # 完整 traceback
    print(traceback.format_exc())

    # 保存到日志
    import logging
    logging.exception("发生错误")
```

---

## 异常链

### raise from

```python
def process_data(data):
    try:
        return int(data)
    except ValueError as e:
        raise RuntimeError("数据处理失败") from e

try:
    process_data("abc")
except RuntimeError as e:
    print(e)           # 数据处理失败
    print(e.__cause__)  # invalid literal for int()...
```

输出：
```
Traceback (most recent call last):
  File "...", line 3, in process_data
    return int(data)
ValueError: invalid literal for int() with base 10: 'abc'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "...", line 7, in <module>
    process_data("abc")
  File "...", line 5, in process_data
    raise RuntimeError("数据处理失败") from e
RuntimeError: 数据处理失败
```

### 隐式异常链

```python
try:
    1 / 0
except:
    raise ValueError("处理时出错")
    # 自动记录原因：__context__
```

### 禁用异常链

```python
try:
    1 / 0
except:
    raise ValueError("新错误") from None
    # 不显示原始异常
```

---

## 自定义异常

### 基本定义

```python
class MyError(Exception):
    """自定义异常"""
    pass

class ValidationError(Exception):
    """验证错误"""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

# 使用
try:
    raise ValidationError("email", "格式无效")
except ValidationError as e:
    print(f"字段 {e.field}: {e.message}")
```

### 异常层级

```python
class AppError(Exception):
    """应用基础异常"""
    pass

class DatabaseError(AppError):
    """数据库错误"""
    pass

class NetworkError(AppError):
    """网络错误"""
    pass

class AuthError(AppError):
    """认证错误"""
    pass

# 捕获所有应用异常
try:
    operation()
except AppError as e:
    handle_app_error(e)
```

---

## 调试技巧

### 使用 pdb

```python
import pdb

def buggy_function():
    x = 1
    pdb.set_trace()  # 在这里暂停
    y = x + "2"  # 这行会出错
    return y

buggy_function()
```

### breakpoint()（Python 3.7+）

```python
def debug_me():
    x = 1
    breakpoint()  # 更简洁的方式
    y = x + 2
    return y
```

### pdb 常用命令

| 命令 | 说明 |
|------|------|
| `n` (next) | 下一行 |
| `s` (step) | 进入函数 |
| `c` (continue) | 继续执行 |
| `l` (list) | 显示代码 |
| `p expr` | 打印表达式 |
| `pp expr` | 美化打印 |
| `q` (quit) | 退出 |
| `h` (help) | 帮助 |

---

## 异常最佳实践

### 精确捕获

```python
# 不好：捕获太宽泛
try:
    operation()
except:  # 或 except Exception:
    pass

# 好：精确捕获
try:
    operation()
except SpecificError as e:
    handle(e)
```

### 不要忽略异常

```python
# 不好
try:
    operation()
except SomeError:
    pass  # 沉默地忽略

# 好
try:
    operation()
except SomeError:
    logging.warning("操作失败，使用默认值")
    return default_value
```

### 使用 with 语句

```python
# 不好
f = open("file.txt")
try:
    data = f.read()
finally:
    f.close()

# 好
with open("file.txt") as f:
    data = f.read()
```

### 异常消息要有意义

```python
# 不好
raise ValueError("错误")

# 好
raise ValueError(f"用户年龄 {age} 必须在 0-150 之间")
```

---

## 本节要点

1. **Traceback 阅读**: 从上到下是调用栈，最后是错误
2. **异常层级**: Exception 是大多数异常的基类
3. **try-except-else-finally**: 完整的异常处理结构
4. **raise from**: 显式异常链，保留原因
5. **自定义异常**: 继承 Exception，添加有意义的信息
6. **pdb/breakpoint**: 调试工具

