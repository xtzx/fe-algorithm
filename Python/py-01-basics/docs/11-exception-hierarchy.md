# 异常层次结构

> 理解 Python 异常体系，选择正确的异常类型

## 异常层次结构图

```
BaseException
├── SystemExit                 # sys.exit() 触发
├── KeyboardInterrupt          # Ctrl+C 触发
├── GeneratorExit              # 生成器关闭时触发
└── Exception                  # 所有常规异常的基类
    ├── StopIteration          # 迭代器耗尽
    ├── ArithmeticError        # 算术异常基类
    │   ├── ZeroDivisionError  # 除零错误
    │   ├── OverflowError      # 数值溢出
    │   └── FloatingPointError # 浮点运算错误
    ├── AssertionError         # assert 失败
    ├── AttributeError         # 属性不存在
    ├── BufferError            # 缓冲区相关错误
    ├── EOFError               # input() 遇到 EOF
    ├── ImportError            # 导入失败
    │   └── ModuleNotFoundError # 模块不存在
    ├── LookupError            # 查找异常基类
    │   ├── IndexError         # 索引越界
    │   └── KeyError           # 键不存在
    ├── MemoryError            # 内存不足
    ├── NameError              # 名称未定义
    │   └── UnboundLocalError  # 局部变量未绑定
    ├── OSError                # 系统相关错误
    │   ├── FileNotFoundError  # 文件不存在
    │   ├── FileExistsError    # 文件已存在
    │   ├── PermissionError    # 权限错误
    │   ├── TimeoutError       # 超时
    │   └── ConnectionError    # 连接错误
    │       ├── ConnectionRefusedError
    │       ├── ConnectionResetError
    │       └── ConnectionAbortedError
    ├── RuntimeError           # 运行时错误
    │   ├── NotImplementedError # 未实现
    │   └── RecursionError     # 递归过深
    ├── SyntaxError            # 语法错误
    │   └── IndentationError   # 缩进错误
    │       └── TabError       # Tab/空格混用
    ├── TypeError              # 类型错误
    ├── ValueError             # 值错误
    │   └── UnicodeError       # Unicode 错误
    └── Warning                # 警告基类
        ├── DeprecationWarning
        ├── FutureWarning
        └── UserWarning
```

---

## BaseException vs Exception

```python
# BaseException: 所有异常的根基类
# Exception: 常规异常的基类（推荐捕获这个）

# ❌ 不要捕获 BaseException，会拦截 Ctrl+C
try:
    while True:
        pass
except BaseException:  # 危险！无法用 Ctrl+C 退出
    pass

# ✅ 捕获 Exception
try:
    risky_operation()
except Exception as e:  # Ctrl+C 仍然有效
    print(f"Error: {e}")
```

**为什么 KeyboardInterrupt 不继承 Exception？**
- 设计意图：用户按 Ctrl+C 应该能中断程序
- 如果继承 Exception，会被 `except Exception` 捕获

---

## 常用异常分类

### 1. 值与类型错误

| 异常 | 触发场景 | 示例 |
|------|---------|------|
| `TypeError` | 类型不匹配 | `"1" + 1` |
| `ValueError` | 值不合法 | `int("abc")` |

```python
# TypeError: 操作数类型不匹配
"hello" + 123  # TypeError: can only concatenate str to str

# ValueError: 类型正确但值不合法
int("abc")     # ValueError: invalid literal for int()
int("3.14")    # ValueError: invalid literal for int()

# 区分两者的技巧
# TypeError: 换个类型就能成功
# ValueError: 换个值（同类型）才能成功
```

### 2. 查找错误

| 异常 | 触发场景 | 示例 |
|------|---------|------|
| `KeyError` | 字典键不存在 | `d["missing"]` |
| `IndexError` | 列表索引越界 | `lst[100]` |
| `AttributeError` | 属性不存在 | `obj.missing` |
| `NameError` | 变量未定义 | `undefined_var` |

```python
# KeyError
d = {"a": 1}
d["b"]  # KeyError: 'b'

# 安全获取
d.get("b")          # None
d.get("b", 0)       # 0（默认值）

# IndexError
lst = [1, 2, 3]
lst[10]  # IndexError: list index out of range

# 安全获取（切片不会报错）
lst[10:]  # []（空列表）

# AttributeError
class Foo:
    pass
Foo().bar  # AttributeError: 'Foo' object has no attribute 'bar'

# 安全获取
getattr(Foo(), "bar", None)  # None
hasattr(Foo(), "bar")        # False
```

### 3. 文件与系统错误

| 异常 | 触发场景 | 示例 |
|------|---------|------|
| `FileNotFoundError` | 文件不存在 | `open("missing.txt")` |
| `PermissionError` | 无权限 | 访问受保护文件 |
| `IsADirectoryError` | 期望文件但是目录 | `open("dir/")` |
| `TimeoutError` | 操作超时 | 网络请求超时 |

```python
from pathlib import Path

# FileNotFoundError
open("not_exists.txt")  # FileNotFoundError

# 安全检查
path = Path("file.txt")
if path.exists():
    content = path.read_text()

# PermissionError
open("/etc/passwd", "w")  # PermissionError (Unix)

# TimeoutError
import socket
s = socket.socket()
s.settimeout(1)
s.connect(("1.2.3.4", 80))  # TimeoutError
```

### 4. 迭代相关

| 异常 | 触发场景 | 示例 |
|------|---------|------|
| `StopIteration` | 迭代器耗尽 | `next(iter([]))` |
| `GeneratorExit` | 生成器被关闭 | `gen.close()` |

```python
# StopIteration（通常不需要手动处理）
it = iter([1, 2])
next(it)  # 1
next(it)  # 2
next(it)  # StopIteration

# 安全获取
next(it, None)  # None（默认值）

# for 循环自动处理 StopIteration
for x in [1, 2, 3]:  # 内部捕获 StopIteration
    print(x)
```

---

## JS 对照：Error 类型 vs Python 异常

| JavaScript | Python | 说明 |
|------------|--------|------|
| `Error` | `Exception` | 基础异常类 |
| `TypeError` | `TypeError` | 类型错误 |
| `ReferenceError` | `NameError` | 变量未定义 |
| `SyntaxError` | `SyntaxError` | 语法错误 |
| `RangeError` | `ValueError` / `IndexError` | 值/索引超范围 |
| 无直接对应 | `KeyError` | 键不存在 |
| 无直接对应 | `AttributeError` | 属性不存在 |
| 无直接对应 | `FileNotFoundError` | 文件不存在 |

```javascript
// JavaScript
try {
    undefinedVar;  // ReferenceError
} catch (e) {
    console.log(e instanceof ReferenceError);  // true
}
```

```python
# Python
try:
    undefined_var  # NameError
except NameError as e:
    print(type(e).__name__)  # NameError
```

---

## 何时捕获何种异常

### 原则：尽量具体

```python
# ❌ 太宽泛
try:
    result = data[key]
except Exception:
    result = default

# ✅ 具体捕获
try:
    result = data[key]
except KeyError:
    result = default
```

### 常见场景与推荐异常

| 场景 | 推荐捕获 |
|------|---------|
| 字典取值 | `KeyError` |
| 列表索引 | `IndexError` |
| 文件操作 | `FileNotFoundError`, `PermissionError`, `OSError` |
| 类型转换 | `ValueError`, `TypeError` |
| JSON 解析 | `json.JSONDecodeError` |
| 网络请求 | `requests.RequestException` (第三方库) |
| 数据库操作 | 库特定异常 |

### 多异常捕获

```python
# 方式一：元组
try:
    value = int(user_input)
except (ValueError, TypeError) as e:
    print(f"Invalid input: {e}")

# 方式二：多个 except
try:
    result = process(data)
except KeyError:
    result = handle_missing_key()
except ValueError:
    result = handle_invalid_value()
except Exception as e:
    # 兜底处理
    log_error(e)
    raise
```

---

## 异常检查实用函数

```python
def is_exception_subclass(exc_type: type, parent: type) -> bool:
    """检查异常类型是否是某个异常的子类"""
    return issubclass(exc_type, parent)

# 示例
print(issubclass(FileNotFoundError, OSError))  # True
print(issubclass(KeyError, LookupError))       # True
print(issubclass(ValueError, TypeError))       # False

# 运行时检查
try:
    raise FileNotFoundError("test")
except OSError:
    print("Caught!")  # FileNotFoundError 是 OSError 子类
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 捕获 `BaseException` | 会拦截 `KeyboardInterrupt` | 捕获 `Exception` |
| 裸 `except:` | 等同于 `except BaseException` | 至少用 `except Exception` |
| 捕获后静默 | 隐藏了错误 | 至少记录日志 |
| 过度宽泛 | 掩盖真实问题 | 捕获具体异常 |

---

## 小结

1. **继承关系决定捕获范围**：捕获父类会捕获所有子类
2. **Exception 是安全的顶级捕获**：不会拦截系统级异常
3. **优先捕获具体异常**：便于定位和处理问题
4. **利用异常层次简化代码**：如 `OSError` 覆盖所有文件相关异常

