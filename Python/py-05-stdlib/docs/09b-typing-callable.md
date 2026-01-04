# Callable 与 ParamSpec

> 函数类型和装饰器的类型安全

## Callable 基础

`Callable` 用于标注可调用对象（函数、方法、类等）的类型。

```python
from typing import Callable

# Callable[[参数类型...], 返回类型]
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def add(x: int, y: int) -> int:
    return x + y

result = apply(add, 1, 2)  # 3
```

### 语法详解

```python
from typing import Callable

# 无参数，返回 int
Callable[[], int]

# 一个 str 参数，返回 bool
Callable[[str], bool]

# 两个参数（int, str），返回 None
Callable[[int, str], None]

# 任意参数，返回 str
Callable[..., str]
```

### 实际示例

```python
from typing import Callable

# 回调函数类型
OnComplete = Callable[[str, int], None]

def process_with_callback(data: str, on_complete: OnComplete) -> None:
    result = len(data)
    on_complete(data, result)

# 使用
def my_callback(data: str, length: int) -> None:
    print(f"Processed {data}, length: {length}")

process_with_callback("hello", my_callback)
```

---

## Callable 的限制

### 问题：丢失参数名

```python
from typing import Callable

# 无法表达参数名
Handler = Callable[[str, int], None]

# 实际函数可能是：
def handler1(name: str, age: int) -> None: ...
def handler2(path: str, size: int) -> None: ...
# 两者类型相同，但语义不同
```

### 问题：无法表达可选参数

```python
from typing import Callable

# 无法表达 def foo(x: int, y: int = 0)
# Callable[[int, int], int] 要求两个参数都必须
```

### 解决方案：Protocol

```python
from typing import Protocol

class Handler(Protocol):
    def __call__(self, name: str, age: int = 0) -> None: ...

def process(handler: Handler) -> None:
    handler("Alice")        # OK，age 有默认值
    handler("Bob", 25)      # OK
```

---

## ParamSpec（Python 3.10+）

`ParamSpec` 解决装饰器中参数签名丢失的问题。

### 问题：传统装饰器

```python
from typing import Callable, TypeVar

T = TypeVar("T")

def logged(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs) -> T:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# 问题：类型检查器不知道 greet 的参数类型
# greet(123)  # 本应报错，但不会
```

### 解决方案：ParamSpec

```python
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def logged(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# 现在类型检查器知道参数类型
greet("Alice")           # OK
greet("Bob", "Hi")       # OK
# greet(123)             # 类型错误！
```

### ParamSpec 的组成

```python
from typing import ParamSpec

P = ParamSpec("P")

# P.args   - 位置参数的类型
# P.kwargs - 关键字参数的类型

def wrapper(*args: P.args, **kwargs: P.kwargs): ...
```

---

## Concatenate

`Concatenate` 用于在 ParamSpec 前添加额外参数。

```python
from typing import Callable, TypeVar, ParamSpec, Concatenate

P = ParamSpec("P")
T = TypeVar("T")

def with_context(
    func: Callable[Concatenate[str, P], T]
) -> Callable[P, T]:
    """添加上下文参数的装饰器"""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        context = get_current_context()  # 自动获取上下文
        return func(context, *args, **kwargs)
    return wrapper

@with_context
def process_data(ctx: str, data: int) -> str:
    return f"[{ctx}] Processing {data}"

# 原函数签名: (ctx: str, data: int) -> str
# 装饰后签名: (data: int) -> str
result = process_data(42)  # ctx 自动注入
```

### 实际示例：依赖注入

```python
from typing import Callable, TypeVar, ParamSpec, Concatenate

P = ParamSpec("P")
T = TypeVar("T")

class Database:
    def query(self, sql: str) -> list:
        return []

def inject_db(
    func: Callable[Concatenate[Database, P], T]
) -> Callable[P, T]:
    """注入数据库连接"""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        db = Database()  # 或从连接池获取
        try:
            return func(db, *args, **kwargs)
        finally:
            pass  # 关闭连接
    return wrapper

@inject_db
def get_users(db: Database, limit: int = 10) -> list:
    return db.query(f"SELECT * FROM users LIMIT {limit}")

# 使用时不需要传 db
users = get_users(limit=5)
```

---

## 完整装饰器示例

### 重试装饰器

```python
from typing import Callable, TypeVar, ParamSpec
import time

P = ParamSpec("P")
T = TypeVar("T")

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """带重试的装饰器"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_error  # type: ignore
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str, timeout: int = 30) -> dict:
    # 网络请求...
    return {}

# 类型安全
result = fetch_data("https://api.example.com")
result = fetch_data("https://api.example.com", timeout=10)
# fetch_data(123)  # 类型错误！
```

### 计时装饰器

```python
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
import time

P = ParamSpec("P")
T = TypeVar("T")

def timed(func: Callable[P, T]) -> Callable[P, T]:
    """计时装饰器，保留签名"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timed
def slow_function(n: int, prefix: str = "") -> str:
    time.sleep(0.1)
    return f"{prefix}{n}"

# 保留了原函数签名
result = slow_function(42)
result = slow_function(42, prefix="Result: ")
```

---

## JS/TS 对照

| Python | TypeScript | 说明 |
|--------|------------|------|
| `Callable[[int], str]` | `(x: number) => string` | 函数类型 |
| `Callable[..., T]` | `(...args: any[]) => T` | 任意参数 |
| `ParamSpec("P")` | 无直接对应 | 参数签名捕获 |
| `P.args, P.kwargs` | 无直接对应 | 参数展开 |
| `Concatenate[X, P]` | 无直接对应 | 参数前置 |

```typescript
// TypeScript 装饰器类型较弱
function logged<T extends (...args: any[]) => any>(
    fn: T
): T {
    return ((...args) => {
        console.log(`Calling ${fn.name}`);
        return fn(...args);
    }) as T;
}
```

```python
# Python 3.10+ 更精确
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

def logged(fn: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 `P.args/P.kwargs` | 签名信息丢失 | 使用 `*args: P.args` |
| Python 版本不够 | ParamSpec 需要 3.10+ | 使用 `typing_extensions` |
| 混淆 TypeVar 和 ParamSpec | 用途不同 | TypeVar 用于返回值 |
| 复杂嵌套 | 类型过于复杂 | 简化或使用 Protocol |

### Python 3.9 兼容

```python
# Python 3.9 使用 typing_extensions
from typing_extensions import ParamSpec, Concatenate

P = ParamSpec("P")
```

