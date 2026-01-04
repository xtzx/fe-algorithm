# 异常链

> 理解 Python 异常的因果关系，优雅地包装和传递异常

## 什么是异常链

当在处理一个异常时引发另一个异常，Python 会保留两个异常的关联，形成**异常链**。

```python
try:
    int("abc")
except ValueError:
    raise RuntimeError("Failed to parse config")

# 输出:
# Traceback (most recent call last):
#   File "...", line 2, in <module>
#     int("abc")
# ValueError: invalid literal for int()...
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "...", line 4, in <module>
#     raise RuntimeError("Failed to parse config")
# RuntimeError: Failed to parse config
```

---

## 显式异常链：raise ... from

### 语法

```python
raise NewException("message") from original_exception
```

### 示例

```python
class ConfigError(Exception):
    """配置错误"""
    pass

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config: {path}") from e

# 使用
try:
    config = load_config("missing.json")
except ConfigError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__cause__}")  # 原始异常
```

### __cause__ 属性

```python
try:
    try:
        int("abc")
    except ValueError as original:
        raise RuntimeError("Conversion failed") from original
except RuntimeError as e:
    print(f"Exception: {e}")
    print(f"Cause: {e.__cause__}")           # ValueError
    print(f"Cause type: {type(e.__cause__)}")  # <class 'ValueError'>
```

---

## 隐式异常链：__context__

当在 `except` 块中抛出新异常（不使用 `from`），Python 自动设置 `__context__`：

```python
try:
    try:
        1 / 0
    except ZeroDivisionError:
        # 在处理过程中又出错
        raise ValueError("Something else went wrong")
except ValueError as e:
    print(f"Exception: {e}")
    print(f"Context: {e.__context__}")  # ZeroDivisionError
```

### __cause__ vs __context__

| 属性 | 设置方式 | 含义 |
|------|---------|------|
| `__cause__` | `raise ... from e` | 显式指定的原因（有意包装）|
| `__context__` | 自动设置 | 处理时的上下文（可能是意外）|

```python
# __cause__: 有意包装
try:
    parse_int("abc")
except ValueError as e:
    raise ConfigError("Invalid config") from e  # 设置 __cause__

# __context__: 意外的链
try:
    1 / 0
except ZeroDivisionError:
    int("abc")  # 意外出错，自动设置 __context__
```

---

## 抑制异常链：raise ... from None

有时不希望显示原始异常，可以抑制异常链：

```python
def get_user(user_id: int):
    try:
        return database.query(f"SELECT * FROM users WHERE id = {user_id}")
    except DatabaseError:
        # 不想暴露内部数据库细节
        raise UserNotFoundError(f"User {user_id} not found") from None

# 输出只有:
# UserNotFoundError: User 123 not found
# （没有 "During handling..." 部分）
```

### 使用场景

```python
# 1. 隐藏内部实现细节
def public_api(data):
    try:
        return internal_complex_operation(data)
    except InternalError:
        raise PublicError("Operation failed") from None

# 2. 简化错误信息
def validate_email(email: str):
    try:
        # 复杂的正则验证
        if not re.match(complex_pattern, email):
            raise re.error("Pattern failed")
    except re.error:
        raise ValueError(f"Invalid email: {email}") from None
```

---

## 读懂异常链

### 输出格式

```python
try:
    try:
        open("missing.txt")
    except FileNotFoundError as e:
        raise RuntimeError("Failed to load data") from e
except RuntimeError:
    raise ValueError("Application error")
```

输出：
```
Traceback (most recent call last):
  File "...", line 3, in <module>
    open("missing.txt")
FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'

The above exception was the direct cause of the following exception:
                    ^^^^^^^^^^^^^^^^^^^
                    使用了 from，所以是 "direct cause"

Traceback (most recent call last):
  File "...", line 5, in <module>
    raise RuntimeError("Failed to load data") from e
RuntimeError: Failed to load data

During handling of the above exception, another exception occurred:
              ^^^^^^^^^^^^^^^^^^^^^^^
              没有 from，所以是 "During handling"

Traceback (most recent call last):
  File "...", line 7, in <module>
    raise ValueError("Application error")
ValueError: Application error
```

### 提示语含义

| 提示语 | 含义 | 对应属性 |
|--------|------|---------|
| "The above exception was the direct cause of..." | 显式链（`from`）| `__cause__` |
| "During handling of the above exception..." | 隐式链 | `__context__` |

---

## 遍历异常链

```python
def print_exception_chain(exc: Exception):
    """打印完整的异常链"""
    chain = []
    current = exc

    while current is not None:
        chain.append(current)
        # 优先使用 __cause__，否则用 __context__
        current = current.__cause__ or current.__context__

    for i, e in enumerate(chain):
        prefix = "  " * i
        print(f"{prefix}[{i}] {type(e).__name__}: {e}")

# 示例
try:
    try:
        int("abc")
    except ValueError as e:
        raise RuntimeError("Parse error") from e
except RuntimeError as e:
    print_exception_chain(e)

# 输出:
# [0] RuntimeError: Parse error
#   [1] ValueError: invalid literal for int()...
```

---

## 实际应用

### 1. API 错误包装

```python
class APIError(Exception):
    """API 层统一异常"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code

def get_user_data(user_id: int) -> dict:
    try:
        user = database.get_user(user_id)
        if not user:
            raise APIError(f"User {user_id} not found", status_code=404) from None
        return user
    except DatabaseConnectionError as e:
        raise APIError("Database unavailable", status_code=503) from e
    except DatabaseError as e:
        raise APIError("Internal error", status_code=500) from e
```

### 2. 数据验证

```python
class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field

def parse_user_input(data: dict) -> User:
    try:
        age = int(data.get("age", ""))
    except ValueError as e:
        raise ValidationError("age", "Must be a number") from e

    try:
        email = validate_email(data.get("email", ""))
    except InvalidEmailError as e:
        raise ValidationError("email", str(e)) from e

    return User(age=age, email=email)
```

### 3. 重试时保留原始错误

```python
class RetryError(Exception):
    """所有重试都失败"""
    pass

def retry(fn, max_attempts: int = 3):
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")

    # 保留最后一次的错误作为原因
    raise RetryError(f"All {max_attempts} attempts failed") from last_error
```

---

## JS 对照

JavaScript 从 ES2022 开始支持 `cause` 选项：

```javascript
// JavaScript (ES2022+)
try {
    JSON.parse(invalidJson);
} catch (e) {
    throw new Error("Config parse failed", { cause: e });
}
```

```python
# Python
try:
    json.loads(invalid_json)
except json.JSONDecodeError as e:
    raise ConfigError("Config parse failed") from e
```

| Python | JavaScript | 说明 |
|--------|------------|------|
| `raise New from orig` | `throw new Error(msg, {cause})` | 显式链 |
| `e.__cause__` | `e.cause` | 获取原因 |
| `raise New from None` | 无直接对应 | 抑制链 |

---

## 最佳实践

### 1. 包装底层异常

```python
# ✅ 提供更有意义的错误信息
try:
    result = complex_operation()
except LowLevelError as e:
    raise HighLevelError(f"Operation failed: {context}") from e
```

### 2. 保留调试信息

```python
# ✅ 使用 from 保留原始错误
try:
    data = fetch_data()
except NetworkError as e:
    raise DataError("Failed to fetch data") from e  # 保留网络错误详情
```

### 3. 适时抑制细节

```python
# ✅ 对外 API 隐藏内部实现
def public_method():
    try:
        return _internal_implementation()
    except _InternalException:
        raise PublicException("Operation failed") from None
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 `from e` | 丢失原始错误信息 | 包装异常时使用 `from` |
| 过度使用 `from None` | 难以调试 | 只在必要时抑制 |
| 忽略异常链 | 不知道根本原因 | 检查 `__cause__` 和 `__context__` |
| 异常链过长 | 输出难以阅读 | 在适当层级处理，避免多层包装 |

