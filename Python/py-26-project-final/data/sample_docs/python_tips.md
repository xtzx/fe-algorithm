# Python 编程技巧

## 类型提示

Python 3.5+ 支持类型提示，提高代码可读性和 IDE 支持：

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_items(items: list[int]) -> dict[str, int]:
    return {"sum": sum(items), "count": len(items)}
```

## 数据类

使用 dataclass 简化数据对象定义：

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
    def distance(self, other: "Point") -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
```

## 上下文管理器

使用 contextmanager 装饰器创建上下文管理器：

```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"耗时: {time.time() - start:.2f}秒")

with timer():
    # 执行一些操作
    time.sleep(1)
```

## 异步编程

使用 async/await 进行异步编程：

```python
import asyncio
import httpx

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    tasks = [
        fetch_data("https://api.example.com/1"),
        fetch_data("https://api.example.com/2"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## 生成器表达式

使用生成器表达式处理大数据集：

```python
# 列表推导式（占用内存）
squares = [x**2 for x in range(1000000)]

# 生成器表达式（惰性求值）
squares_gen = (x**2 for x in range(1000000))

# 按需迭代
for square in squares_gen:
    if square > 100:
        break
```

## 装饰器

创建可复用的装饰器：

```python
import functools
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    # 可能失败的操作
    pass
```


