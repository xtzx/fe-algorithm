# 上下文管理器

> 使用 with 语句优雅地管理资源

## 什么是上下文管理器

上下文管理器是实现了 `__enter__` 和 `__exit__` 方法的对象，用于 `with` 语句。

```python
with open("file.txt") as f:
    content = f.read()
# 文件自动关闭，即使发生异常
```

等价于：

```python
f = open("file.txt")
try:
    content = f.read()
finally:
    f.close()
```

---

## with 语句的执行流程

```python
with expression as variable:
    body
```

执行顺序：
1. 执行 `expression`，得到上下文管理器对象
2. 调用 `__enter__()`，返回值赋给 `variable`
3. 执行 `body`
4. 调用 `__exit__()`（无论是否发生异常）

---

## 实现上下文管理器：类方式

### 基础结构

```python
class MyContext:
    def __enter__(self):
        # 进入时执行
        print("Entering context")
        return self  # 返回值赋给 as 后的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时执行
        print("Exiting context")
        return False  # False: 不抑制异常；True: 抑制异常

# 使用
with MyContext() as ctx:
    print("Inside context")

# 输出:
# Entering context
# Inside context
# Exiting context
```

### __exit__ 参数详解

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    """
    参数:
        exc_type: 异常类型（如 ValueError），无异常时为 None
        exc_val:  异常实例，无异常时为 None
        exc_tb:   traceback 对象，无异常时为 None

    返回值:
        True:  抑制异常，不会向外抛出
        False: 不抑制异常，继续抛出
    """
    if exc_type is not None:
        print(f"Exception: {exc_type.__name__}: {exc_val}")
    return False
```

### 实际示例：计时器

```python
import time

class Timer:
    """计时上下文管理器"""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"Elapsed: {self.elapsed:.4f} seconds")
        return False

# 使用
with Timer() as t:
    # 耗时操作
    sum(range(1000000))

print(f"Total time: {t.elapsed:.4f}s")
```

### 实际示例：数据库事务

```python
class Transaction:
    """数据库事务管理器"""

    def __init__(self, connection):
        self.conn = connection

    def __enter__(self):
        self.conn.begin()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 发生异常，回滚
            self.conn.rollback()
            print(f"Transaction rolled back due to: {exc_val}")
        else:
            # 无异常，提交
            self.conn.commit()
            print("Transaction committed")
        return False  # 不抑制异常

# 使用
with Transaction(db_connection) as conn:
    conn.execute("INSERT INTO users VALUES (...)")
    conn.execute("UPDATE accounts SET ...")
    # 如果任何操作失败，整个事务回滚
```

### 实际示例：临时目录

```python
import os
import tempfile
import shutil

class TempDirectory:
    """临时目录，退出时自动删除"""

    def __init__(self, prefix: str = "tmp_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return self.path

    def __exit__(self, *args):
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path)
        return False

# 使用
with TempDirectory(prefix="myapp_") as tmpdir:
    # 在临时目录中工作
    with open(os.path.join(tmpdir, "test.txt"), "w") as f:
        f.write("temporary data")
# 退出后临时目录被删除
```

---

## 实现上下文管理器：contextlib

### @contextmanager 装饰器

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    # __enter__ 部分
    print("Entering")
    resource = acquire_resource()
    try:
        yield resource  # 返回给 as 变量
    finally:
        # __exit__ 部分
        release_resource(resource)
        print("Exiting")

# 使用
with my_context() as res:
    use(res)
```

### 计时器（简化版）

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str = ""):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s" if name else f"{elapsed:.4f}s")

# 使用
with timer("Data processing"):
    process_data()
```

### 临时修改

```python
from contextlib import contextmanager
import os

@contextmanager
def temp_env(key: str, value: str):
    """临时设置环境变量"""
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[key]
        else:
            os.environ[key] = old_value

# 使用
with temp_env("DEBUG", "true"):
    # 在这里 DEBUG=true
    run_in_debug_mode()
# 退出后恢复原值
```

### 锁管理

```python
from contextlib import contextmanager
import threading

@contextmanager
def locked(lock: threading.Lock):
    """获取锁的上下文管理器"""
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# 使用
lock = threading.Lock()
with locked(lock):
    # 临界区
    modify_shared_resource()
```

---

## contextlib 常用工具

### suppress - 抑制特定异常

```python
from contextlib import suppress

# 忽略 FileNotFoundError
with suppress(FileNotFoundError):
    os.remove("maybe_exists.txt")

# 等价于
try:
    os.remove("maybe_exists.txt")
except FileNotFoundError:
    pass

# 忽略多个异常
with suppress(FileNotFoundError, PermissionError):
    os.remove("file.txt")
```

### closing - 确保调用 close()

```python
from contextlib import closing
from urllib.request import urlopen

# 确保 response 被关闭
with closing(urlopen("https://example.com")) as response:
    html = response.read()

# 用于没有实现 __enter__/__exit__ 但有 close() 的对象
class Resource:
    def close(self):
        print("Resource closed")

with closing(Resource()) as r:
    use(r)
# 输出: Resource closed
```

### nullcontext - 空上下文

```python
from contextlib import nullcontext

def process(file_path: str | None = None):
    # 条件性使用上下文管理器
    if file_path:
        cm = open(file_path, "w")
    else:
        cm = nullcontext(sys.stdout)

    with cm as output:
        output.write("Hello")

# file_path=None 时写到 stdout
# file_path 有值时写到文件
```

### ExitStack - 动态管理多个上下文

```python
from contextlib import ExitStack

def process_files(file_paths: list[str]):
    with ExitStack() as stack:
        # 动态打开多个文件
        files = [
            stack.enter_context(open(path))
            for path in file_paths
        ]

        # 处理所有文件
        for f in files:
            process(f)
    # 所有文件自动关闭

# 条件性添加上下文
with ExitStack() as stack:
    if need_lock:
        stack.enter_context(lock)
    if need_transaction:
        stack.enter_context(transaction)

    do_work()
```

### redirect_stdout / redirect_stderr

```python
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# 捕获 stdout
buffer = StringIO()
with redirect_stdout(buffer):
    print("Hello")
    print("World")

output = buffer.getvalue()
print(f"Captured: {output!r}")  # Captured: 'Hello\nWorld\n'

# 重定向到文件
with open("output.log", "w") as f:
    with redirect_stdout(f):
        print("This goes to file")
```

---

## 异步上下文管理器

### async with 语法

```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

# 使用
async def main():
    async with AsyncResource() as resource:
        await resource.do_something()
```

### @asynccontextmanager

```python
from contextlib import asynccontextmanager
import aiohttp

@asynccontextmanager
async def http_session():
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()

# 使用
async def fetch_data():
    async with http_session() as session:
        async with session.get("https://api.example.com") as response:
            return await response.json()
```

---

## JS 对照

JavaScript 目前没有原生的上下文管理器，但 TC39 提案中有 `using` 声明：

```javascript
// JavaScript (Stage 3 提案)
{
    using file = openFile("data.txt");
    // 使用 file
}  // 自动调用 file[Symbol.dispose]()

// 异步版本
{
    await using conn = await getConnection();
    // 使用 conn
}  // 自动调用 await conn[Symbol.asyncDispose]()
```

```python
# Python
with open("data.txt") as file:
    # 使用 file
# 自动调用 file.__exit__()

# 异步版本
async with get_connection() as conn:
    # 使用 conn
# 自动调用 await conn.__aexit__()
```

| Python | JavaScript (提案) | 说明 |
|--------|------------------|------|
| `with x as y:` | `using y = x;` | 资源管理 |
| `async with x as y:` | `await using y = x;` | 异步资源 |
| `__enter__` / `__exit__` | `[Symbol.dispose]` | 协议方法 |
| `contextlib` | 无内置库 | 工具函数 |

---

## 常见使用场景

| 场景 | 示例 |
|------|------|
| 文件操作 | `with open(path) as f:` |
| 网络连接 | `with socket.create_connection() as sock:` |
| 数据库连接 | `with connection.cursor() as cursor:` |
| 锁 | `with threading.Lock():` |
| 临时修改 | 环境变量、工作目录、配置 |
| 计时 | 性能测量 |
| 抑制异常 | `with suppress(Exception):` |
| 重定向输出 | `with redirect_stdout(f):` |

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| `__exit__` 返回 True | 会抑制所有异常 | 只在需要时返回 True |
| 忘记 `return False` | 隐式返回 None（等于 False）| 显式返回以提高可读性 |
| yield 后没有 finally | 异常时不会清理 | @contextmanager 中用 try-finally |
| 在 __enter__ 中出错 | __exit__ 不会被调用 | 确保 __enter__ 的原子性 |
| 嵌套过深 | 代码难以阅读 | 使用 ExitStack |

---

## 小结

1. **with 语句**：简化资源管理，确保清理
2. **类方式**：实现 `__enter__` 和 `__exit__`
3. **装饰器方式**：使用 `@contextmanager`
4. **contextlib 工具**：`suppress`、`closing`、`ExitStack` 等
5. **异步支持**：`async with` 和 `@asynccontextmanager`

