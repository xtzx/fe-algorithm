# 04. 装饰器（Decorator）

## 🎯 本节目标

- 理解装饰器的原理和语法
- 掌握各种装饰器模式
- 使用 functools.wraps 保留元信息
- 实现实用的装饰器

---

## 📝 什么是装饰器

装饰器（Decorator）是一种**修改或增强函数功能**的方式，而不改变函数本身的定义。

### 语法糖

```python
@decorator
def my_function():
    pass

# 等价于
def my_function():
    pass
my_function = decorator(my_function)
```

### 基本示例

```python
def timer(func):
    """计时装饰器"""
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result

    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()
# slow_function took 1.00s
# 完成
```

---

## 🔧 基础装饰器

### 无参数装饰器

```python
def uppercase(func):
    """将返回值转为大写"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return str(result).upper()
    return wrapper

@uppercase
def greet(name):
    return f"Hello, {name}"

print(greet("Alice"))  # HELLO, ALICE
```

### 保留函数元信息

```python
from functools import wraps

def timer(func):
    @wraps(func)  # ✅ 保留原函数的元信息
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def my_function():
    """这是一个测试函数"""
    pass

print(my_function.__name__)   # my_function（不是 wrapper）
print(my_function.__doc__)   # 这是一个测试函数
```

**不使用 @wraps 的问题**：
```python
def timer(func):
    def wrapper(*args, **kwargs):
        # ...
        return result
    return wrapper

@timer
def my_function():
    """文档"""
    pass

print(my_function.__name__)  # wrapper ❌
print(my_function.__doc__)   # None ❌
```

---

## 🎛️ 带参数的装饰器

需要三层嵌套函数。

### 基本结构

```python
def decorator_with_args(arg1, arg2):
    """装饰器工厂"""
    def decorator(func):
        """真正的装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 可以使用 arg1, arg2
            print(f"装饰器参数: {arg1}, {arg2}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_with_args("param1", "param2")
def my_function():
    pass
```

### 重试装饰器

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"尝试 {attempt + 1} 失败，{delay}秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("随机失败")
    return "成功"
```

### 权限检查装饰器

```python
def require_role(*allowed_roles):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 假设第一个参数是 user
            user = args[0] if args else kwargs.get("user")
            if user.get("role") not in allowed_roles:
                raise PermissionError(f"需要角色: {allowed_roles}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_role("admin", "moderator")
def delete_post(user, post_id):
    return f"删除帖子 {post_id}"

user = {"name": "Alice", "role": "admin"}
delete_post(user, 123)  # ✅
```

---

## 🔄 多个装饰器

多个装饰器从下到上执行。

### 执行顺序

```python
def decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器1：前")
        result = func(*args, **kwargs)
        print("装饰器1：后")
        return result
    return wrapper

def decorator2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器2：前")
        result = func(*args, **kwargs)
        print("装饰器2：后")
        return result
    return wrapper

@decorator1
@decorator2
def my_function():
    print("函数执行")

my_function()
# 装饰器1：前
# 装饰器2：前
# 函数执行
# 装饰器2：后
# 装饰器1：后
```

**等价于**：
```python
my_function = decorator1(decorator2(my_function))
```

---

## 🏛️ 类装饰器

类也可以作为装饰器。

### 基本类装饰器

```python
class Timer:
    """计时类装饰器"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        import time
        start = time.time()
        result = self.func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{self.func.__name__} took {elapsed:.2f}s")
        return result

@Timer
def slow_function():
    time.sleep(1)
```

### 带状态的类装饰器

```python
class Counter:
    """计数装饰器"""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 被调用了 {self.count} 次")
        return self.func(*args, **kwargs)

@Counter
def my_function():
    pass

my_function()  # my_function 被调用了 1 次
my_function()  # my_function 被调用了 2 次
```

---

## 🎨 装饰类的装饰器

装饰器也可以装饰类。

### 单例模式

```python
def singleton(cls):
    """单例装饰器"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Database:
    def __init__(self):
        print("初始化数据库连接")

db1 = Database()  # 初始化数据库连接
db2 = Database()  # 不打印（返回已有实例）
print(db1 is db2)  # True
```

### 添加方法

```python
def add_methods(**methods):
    """给类添加方法的装饰器"""
    def decorator(cls):
        for name, method in methods.items():
            setattr(cls, name, method)
        return cls
    return decorator

@add_methods(
    greet=lambda self: f"Hello, I'm {self.name}",
    age_in_days=lambda self: self.age * 365
)
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 25)
print(p.greet())        # Hello, I'm Alice
print(p.age_in_days())  # 9125
```

---

## 🛠️ 常用内置装饰器

### @property

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径必须 >= 0")
        self._radius = value

c = Circle(5)
print(c.radius)    # 5
c.radius = 10     # 调用 setter
```

### @classmethod

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2024 - birth_year
        return cls(name, age)

p = Person.from_birth_year("Alice", 1999)
```

### @staticmethod

```python
class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y

result = MathUtils.add(3, 5)  # 8
```

### @dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

---

## 🎯 实战装饰器

### 1. 缓存装饰器

```python
from functools import wraps

def cache(func):
    """简单缓存装饰器"""
    cache_dict = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache_dict:
            return cache_dict[key]
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result

    return wrapper

@cache
def expensive_function(n):
    print(f"计算 {n}...")
    return n ** 2

print(expensive_function(5))  # 计算 5... 25
print(expensive_function(5))  # 25（从缓存返回）
```

### 2. 参数验证装饰器

```python
def validate_types(**types):
    """类型验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 验证位置参数
            for i, (arg, expected_type) in enumerate(zip(args, types.values())):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"参数 {i} 应该是 {expected_type.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

create_user("Alice", 25)  # ✅
# create_user("Alice", "25")  # ❌ TypeError
```

### 3. 日志装饰器

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_call(func):
    """记录函数调用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"调用 {func.__name__}，参数: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} 返回: {result}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} 出错: {e}")
            raise
    return wrapper

@log_call
def divide(a, b):
    return a / b
```

### 4. 限流装饰器

```python
import time
from functools import wraps
from collections import defaultdict

def rate_limit(max_calls=5, period=60):
    """限流装饰器"""
    def decorator(func):
        calls = defaultdict(list)

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = id(args)  # 简单的键

            # 清理过期记录
            calls[key] = [t for t in calls[key] if now - t < period]

            if len(calls[key]) >= max_calls:
                raise Exception(f"超过限流：{max_calls} 次/{period}秒")

            calls[key].append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator

@rate_limit(max_calls=3, period=10)
def api_call():
    return "API 响应"
```

---

## ✅ 本节要点

1. 装饰器是修改函数功能的语法糖
2. `@wraps` 保留原函数的元信息
3. 带参数装饰器需要三层嵌套
4. 多个装饰器从下到上执行
5. 类也可以作为装饰器
6. 装饰器可以装饰类
7. 常用装饰器：timer、retry、cache、validate

