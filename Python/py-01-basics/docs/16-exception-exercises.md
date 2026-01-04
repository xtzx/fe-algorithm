# 异常处理练习题

## 基础练习

### 练习 1：基本异常捕获

捕获用户输入的数字转换错误。

```python
def safe_int_input(prompt: str) -> int:
    """
    安全地获取用户输入的整数
    - 如果输入无效，提示重新输入
    - 最多尝试 3 次
    """
    # 你的代码
    pass

# 测试
# result = safe_int_input("Enter a number: ")
```

<details>
<summary>参考答案</summary>

```python
def safe_int_input(prompt: str, max_attempts: int = 3) -> int:
    for attempt in range(max_attempts):
        try:
            return int(input(prompt))
        except ValueError:
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"Invalid input. {remaining} attempts remaining.")
            else:
                raise ValueError("Max attempts exceeded")
```
</details>

---

### 练习 2：多异常处理

实现一个安全的字典取值函数。

```python
def safe_get(data: dict, key: str, convert_to: type = str):
    """
    安全地从字典获取值并转换类型
    - KeyError: 返回 None
    - TypeError/ValueError: 返回原始值
    """
    # 你的代码
    pass

# 测试
data = {"age": "25", "name": "Alice", "score": "invalid"}
assert safe_get(data, "age", int) == 25
assert safe_get(data, "missing", int) is None
assert safe_get(data, "score", int) == "invalid"
```

<details>
<summary>参考答案</summary>

```python
def safe_get(data: dict, key: str, convert_to: type = str):
    try:
        value = data[key]
        return convert_to(value)
    except KeyError:
        return None
    except (TypeError, ValueError):
        return data.get(key)
```
</details>

---

### 练习 3：finally 的作用

解释以下代码的输出：

```python
def demo():
    try:
        print("A")
        return "B"
    except Exception:
        print("C")
        return "D"
    finally:
        print("E")

result = demo()
print(f"Result: {result}")
```

<details>
<summary>答案</summary>

输出：
```
A
E
Result: B
```

解释：
1. 执行 try 块，打印 "A"
2. 准备返回 "B"，但先执行 finally
3. 执行 finally，打印 "E"
4. 返回 "B"
5. 打印 "Result: B"

注意：没有异常，所以 except 块不执行。
</details>

---

### 练习 4：else 子句

重构以下代码，使用 else 子句：

```python
# 原始代码
def load_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
            # 验证数据
            if "version" not in data:
                raise ValueError("Missing version")
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
```

<details>
<summary>参考答案</summary>

```python
def load_json(path: str) -> dict:
    try:
        f = open(path)
    except FileNotFoundError:
        return {}

    try:
        with f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    else:
        # 验证数据（不会被上面的 except 捕获）
        if "version" not in data:
            raise ValueError("Missing version")
        return data
```
</details>

---

### 练习 5：异常信息提取

编写函数提取异常的完整信息：

```python
def get_exception_info(exc: Exception) -> dict:
    """
    返回:
    {
        "type": 异常类型名,
        "message": 异常消息,
        "cause": 原因异常（如果有）,
        "context": 上下文异常（如果有）
    }
    """
    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
def get_exception_info(exc: Exception) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "cause": get_exception_info(exc.__cause__) if exc.__cause__ else None,
        "context": get_exception_info(exc.__context__) if exc.__context__ else None,
    }
```
</details>

---

## 进阶练习

### 练习 6：自定义异常体系

设计一个文件处理的异常体系：

```python
# 要求：
# 1. FileError 基类
# 2. FileNotFoundError（继承自 FileError）
# 3. FilePermissionError（继承自 FileError）
# 4. FileFormatError（继承自 FileError，包含 line_number 属性）

# 你的代码

# 测试
try:
    raise FileFormatError("Invalid syntax", line_number=42)
except FileError as e:
    print(f"Caught: {e}")
```

<details>
<summary>参考答案</summary>

```python
class FileError(Exception):
    """文件处理异常基类"""
    pass

class FileNotFoundError(FileError):
    """文件不存在"""
    pass

class FilePermissionError(FileError):
    """权限不足"""
    pass

class FileFormatError(FileError):
    """文件格式错误"""
    def __init__(self, message: str, line_number: int | None = None):
        super().__init__(message)
        self.line_number = line_number

    def __str__(self):
        if self.line_number:
            return f"{super().__str__()} (line {self.line_number})"
        return super().__str__()
```
</details>

---

### 练习 7：异常链使用

实现配置加载函数，正确使用异常链：

```python
class ConfigError(Exception):
    pass

def load_config(path: str) -> dict:
    """
    加载配置文件
    - 文件不存在：抛出 ConfigError，保留原因
    - JSON 解析失败：抛出 ConfigError，保留原因
    - 返回配置字典
    """
    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
import json

class ConfigError(Exception):
    pass

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {path}") from e
```
</details>

---

### 练习 8：上下文管理器 - 类方式

实现一个日志上下文管理器：

```python
class LogContext:
    """
    记录代码块的执行日志
    - 进入时记录开始
    - 退出时记录结束（包括是否成功）
    - 发生异常时记录异常信息
    """

    def __init__(self, name: str):
        self.name = name

    # 实现 __enter__ 和 __exit__

# 使用
with LogContext("data_processing"):
    process_data()
# 输出:
# [START] data_processing
# [END] data_processing (success)

# 或发生异常时:
# [START] data_processing
# [ERROR] data_processing: ValueError: invalid data
# [END] data_processing (failed)
```

<details>
<summary>参考答案</summary>

```python
class LogContext:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"[ERROR] {self.name}: {exc_type.__name__}: {exc_val}")
            print(f"[END] {self.name} (failed)")
        else:
            print(f"[END] {self.name} (success)")
        return False  # 不抑制异常
```
</details>

---

### 练习 9：上下文管理器 - 装饰器方式

使用 `@contextmanager` 实现临时工作目录切换：

```python
from contextlib import contextmanager

@contextmanager
def temp_chdir(path: str):
    """
    临时切换工作目录
    - 进入时切换到指定目录
    - 退出时恢复原目录
    - 即使发生异常也要恢复
    """
    # 你的代码
    pass

# 使用
print(os.getcwd())  # /home/user
with temp_chdir("/tmp"):
    print(os.getcwd())  # /tmp
print(os.getcwd())  # /home/user
```

<details>
<summary>参考答案</summary>

```python
import os
from contextlib import contextmanager

@contextmanager
def temp_chdir(path: str):
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)
```
</details>

---

### 练习 10：ExitStack 使用

使用 ExitStack 处理多个文件：

```python
def merge_files(input_paths: list[str], output_path: str):
    """
    合并多个文件到一个输出文件
    - 使用 ExitStack 管理所有文件句柄
    - 确保所有文件都被正确关闭
    """
    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
from contextlib import ExitStack

def merge_files(input_paths: list[str], output_path: str):
    with ExitStack() as stack:
        # 打开所有输入文件
        input_files = [
            stack.enter_context(open(path, 'r'))
            for path in input_paths
        ]
        # 打开输出文件
        output_file = stack.enter_context(open(output_path, 'w'))

        # 合并内容
        for f in input_files:
            output_file.write(f.read())
            output_file.write('\n')
```
</details>

---

## 挑战练习

### 挑战 1：完整的应用异常体系

设计一个 Web API 的完整异常体系：

```python
# 要求：
# 1. APIError 基类，包含 status_code, code, message
# 2. ValidationError (400)
# 3. AuthenticationError (401)
# 4. AuthorizationError (403)
# 5. NotFoundError (404)
# 6. ConflictError (409)
# 7. InternalError (500)
# 8. to_response() 方法返回 API 响应格式
```

<details>
<summary>参考答案</summary>

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class APIError(Exception):
    message: str
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    details: dict[str, Any] | None = None

    def __post_init__(self):
        super().__init__(self.message)
        self.timestamp = datetime.now()

    def to_response(self) -> dict:
        response = {
            "error": {
                "code": self.code,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
            }
        }
        if self.details:
            response["error"]["details"] = self.details
        return response

class ValidationError(APIError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 400, "VALIDATION_ERROR", details)

class AuthenticationError(APIError):
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, 401, "AUTHENTICATION_ERROR")

class AuthorizationError(APIError):
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, 403, "AUTHORIZATION_ERROR")

class NotFoundError(APIError):
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            f"{resource} not found: {resource_id}",
            404,
            "NOT_FOUND",
            {"resource": resource, "id": resource_id}
        )

class ConflictError(APIError):
    def __init__(self, message: str):
        super().__init__(message, 409, "CONFLICT")

class InternalError(APIError):
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, 500, "INTERNAL_ERROR")
```
</details>

---

### 挑战 2：带重试的上下文管理器

```python
@contextmanager
def retry_context(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    重试上下文管理器
    - 发生指定异常时自动重试
    - 支持配置重试次数和延迟
    - 所有重试失败后抛出最后一个异常
    """
    # 你的代码
    pass

# 使用
with retry_context(max_retries=3, delay=0.5, exceptions=(ConnectionError,)):
    response = requests.get(url)
```

<details>
<summary>参考答案</summary>

```python
import time
from contextlib import contextmanager

class RetryFailed(Exception):
    """所有重试都失败"""
    pass

@contextmanager
def retry_context(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    last_exception = None

    for attempt in range(max_retries):
        try:
            yield attempt  # 返回当前尝试次数
            return  # 成功，退出
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
            # 继续下一次尝试

    # 所有重试失败
    raise RetryFailed(f"All {max_retries} attempts failed") from last_exception

# 注意：这个实现有限制，因为 yield 只能执行一次
# 更好的方式是使用装饰器或显式循环
```

更好的实现（使用装饰器）：

```python
from functools import wraps

def retry(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise RetryFailed(f"All {max_retries} attempts failed") from last_exception
        return wrapper
    return decorator

@retry(max_retries=3, exceptions=(ConnectionError,))
def fetch_data():
    return requests.get(url)
```
</details>

---

### 挑战 3：异常处理与日志结合

```python
import logging

def logged_exceptions(logger: logging.Logger):
    """
    装饰器：自动记录异常日志
    - 捕获所有异常并记录
    - 记录完整的异常链
    - 重新抛出异常
    """
    # 你的代码
    pass

# 使用
logger = logging.getLogger(__name__)

@logged_exceptions(logger)
def risky_operation():
    ...
```

<details>
<summary>参考答案</summary>

```python
import logging
import traceback
from functools import wraps

def logged_exceptions(logger: logging.Logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录异常信息
                logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True  # 包含完整 traceback
                )

                # 记录异常链
                if e.__cause__:
                    logger.error(f"Caused by: {type(e.__cause__).__name__}: {e.__cause__}")
                if e.__context__ and e.__context__ is not e.__cause__:
                    logger.error(f"Context: {type(e.__context__).__name__}: {e.__context__}")

                raise
        return wrapper
    return decorator
```
</details>

---

## 面试题

### 1. try-except-else-finally 的执行顺序是什么？

<details>
<summary>答案</summary>

- **无异常**：try → else → finally
- **有异常被捕获**：try → except → finally
- **有异常未被捕获**：try → finally → 向上抛出

else 子句只在 try 块成功完成且无异常时执行。
finally 总是执行，即使 try/except 中有 return。
</details>

### 2. raise ... from e 和直接 raise 的区别？

<details>
<summary>答案</summary>

- `raise NewError from e`：显式设置 `__cause__`，表示有意的异常包装
- 在 except 中直接 `raise NewError`：自动设置 `__context__`，表示处理时意外发生

输出区别：
- from：显示 "The above exception was the direct cause of..."
- 无 from：显示 "During handling of the above exception..."
</details>

### 3. 什么时候用 except Exception vs except BaseException？

<details>
<summary>答案</summary>

- `except Exception`：捕获所有常规异常，推荐使用
- `except BaseException`：捕获所有异常，包括 `KeyboardInterrupt`、`SystemExit`

通常应该使用 `except Exception`，因为：
- 用户应该能通过 Ctrl+C 中断程序
- `sys.exit()` 应该能正常退出
</details>

### 4. 上下文管理器的 __exit__ 返回 True 有什么作用？

<details>
<summary>答案</summary>

返回 True 会抑制异常，异常不会向外传播。

```python
class SuppressError:
    def __enter__(self): pass
    def __exit__(self, *args): return True

with SuppressError():
    raise ValueError("This won't propagate")
print("Continues normally")
```

应谨慎使用，可能隐藏真正的错误。
</details>

### 5. 如何实现一个可重用的资源管理器？

<details>
<summary>答案</summary>

使用 `@contextmanager` 装饰器：

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire()
    try:
        yield resource
    finally:
        release(resource)
```

或实现类：

```python
class ManagedResource:
    def __enter__(self):
        self.resource = acquire()
        return self.resource

    def __exit__(self, *args):
        release(self.resource)
        return False
```
</details>

