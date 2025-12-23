# 🐍 04 - Python 进阶

> 掌握 Python 进阶特性，写出更优雅的代码

---

## 目录

1. [列表/字典/集合推导式](#1-推导式)
2. [装饰器](#2-装饰器)
3. [生成器与迭代器](#3-生成器与迭代器)
4. [面向对象编程](#4-面向对象编程)
5. [异步编程基础](#5-异步编程基础)
6. [类型注解](#6-类型注解)
7. [常用内置函数](#7-常用内置函数)
8. [练习题](#8-练习题)

---

## 1. 推导式

### 1.1 列表推导式

```python
# 基本语法: [expression for item in iterable if condition]

# 普通写法
squares = []
for x in range(10):
    squares.append(x ** 2)

# 列表推导式
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件
evens = [x for x in range(20) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# 多重条件
nums = [x for x in range(50) if x % 2 == 0 if x % 3 == 0]
print(nums)  # [0, 6, 12, 18, 24, 30, 36, 42, 48]

# if-else 表达式
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(labels)  # ['even', 'odd', 'even', 'odd', 'even']

# 嵌套循环
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 创建二维数组
grid = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(grid)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```

### 1.2 字典推导式

```python
# 基本语法: {key_expr: value_expr for item in iterable if condition}

# 创建字典
squares = {x: x**2 for x in range(6)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 反转字典
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {v: k for k, v in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# 过滤字典
scores = {"Alice": 85, "Bob": 62, "Charlie": 91, "David": 58}
passed = {k: v for k, v in scores.items() if v >= 60}
print(passed)  # {'Alice': 85, 'Bob': 62, 'Charlie': 91}

# 从两个列表创建字典
keys = ["name", "age", "city"]
values = ["Alice", 25, "NYC"]
person = {k: v for k, v in zip(keys, values)}
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'NYC'}
```

### 1.3 集合推导式

```python
# 基本语法: {expression for item in iterable if condition}

# 创建集合
squares = {x**2 for x in range(-5, 6)}
print(squares)  # {0, 1, 4, 9, 16, 25}

# 去重并转换
words = ["Hello", "HELLO", "hello", "World", "world"]
unique_lower = {w.lower() for w in words}
print(unique_lower)  # {'hello', 'world'}
```

---

## 2. 装饰器

### 2.1 基本概念

```python
# 装饰器是一个函数，接收一个函数，返回一个新函数

# 没有装饰器的写法
def my_function():
    print("Hello!")

def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

my_function = decorator(my_function)
my_function()
# 输出:
# Before
# Hello!
# After
```

### 2.2 使用 @ 语法

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator  # 等价于 say_hello = my_decorator(say_hello)
def say_hello():
    print("Hello!")

say_hello()
# 输出:
# Before function call
# Hello!
# After function call
```

### 2.3 带参数的函数

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Done")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

result = add(3, 5)
print(f"Result: {result}")
# 输出:
# Calling add
# Done
# Result: 8
```

### 2.4 实用装饰器示例

```python
import time
from functools import wraps

# 计时装饰器
def timer(func):
    @wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function 耗时: 1.0012s

# 缓存装饰器
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # 瞬间计算出来

# 重试装饰器
def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=3)
def risky_operation():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success"
```

---

## 3. 生成器与迭代器

### 3.1 迭代器

```python
# 迭代器是实现了 __iter__ 和 __next__ 方法的对象

# 列表是可迭代对象，但不是迭代器
my_list = [1, 2, 3]
my_iter = iter(my_list)  # 转换为迭代器

print(next(my_iter))  # 1
print(next(my_iter))  # 2
print(next(my_iter))  # 3
# print(next(my_iter))  # StopIteration 异常

# 自定义迭代器
class Counter:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

for num in Counter(1, 5):
    print(num)  # 1, 2, 3, 4
```

### 3.2 生成器函数

```python
# 生成器是一种特殊的迭代器，使用 yield 关键字

def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

# 使用
for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5

# 生成器是惰性的，只在需要时计算
gen = count_up_to(1000000)  # 不会立即生成所有数
print(next(gen))  # 1
print(next(gen))  # 2

# 实用示例：读取大文件
def read_large_file(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()

# 斐波那契生成器
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci_gen()
for _ in range(10):
    print(next(fib), end=" ")  # 0 1 1 2 3 5 8 13 21 34
```

### 3.3 生成器表达式

```python
# 类似列表推导式，但用圆括号
squares_list = [x**2 for x in range(1000000)]  # 立即创建列表，占用大量内存
squares_gen = (x**2 for x in range(1000000))   # 生成器，几乎不占内存

# 生成器表达式可以直接作为函数参数
total = sum(x**2 for x in range(100))
print(total)  # 328350

# 查找满足条件的第一个元素
numbers = [1, 4, 6, 8, 11, 15]
first_even = next((x for x in numbers if x % 2 == 0), None)
print(first_even)  # 4
```

---

## 4. 面向对象编程

### 4.1 类的定义

```python
class Dog:
    # 类属性（所有实例共享）
    species = "Canis familiaris"

    # 构造方法
    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age

    # 实例方法
    def bark(self):
        return f"{self.name} says Woof!"

    def get_info(self):
        return f"{self.name} is {self.age} years old"

    # 特殊方法：字符串表示
    def __str__(self):
        return f"Dog({self.name}, {self.age})"

    def __repr__(self):
        return f"Dog(name='{self.name}', age={self.age})"

# 创建实例
my_dog = Dog("Buddy", 3)
print(my_dog.name)       # Buddy
print(my_dog.bark())     # Buddy says Woof!
print(my_dog)            # Dog(Buddy, 3)
print(Dog.species)       # Canis familiaris
```

### 4.2 继承

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现此方法")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# 使用
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())
# Buddy says Woof!
# Whiskers says Meow!
```

### 4.3 类方法和静态方法

```python
class MyClass:
    class_variable = 0

    def __init__(self, value):
        self.value = value

    # 实例方法：第一个参数是 self
    def instance_method(self):
        return f"Instance method, value = {self.value}"

    # 类方法：第一个参数是 cls
    @classmethod
    def class_method(cls):
        return f"Class method, class_variable = {cls.class_variable}"

    # 静态方法：不需要 self 或 cls
    @staticmethod
    def static_method(x, y):
        return x + y

# 使用
obj = MyClass(10)
print(obj.instance_method())    # Instance method, value = 10
print(MyClass.class_method())   # Class method, class_variable = 0
print(MyClass.static_method(3, 5))  # 8
```

### 4.4 数据类（Python 3.7+）

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str = ""  # 默认值
    hobbies: List[str] = field(default_factory=list)  # 可变默认值

# 自动生成 __init__, __repr__, __eq__ 等方法
alice = Person("Alice", 25, "alice@example.com")
bob = Person("Bob", 30)

print(alice)  # Person(name='Alice', age=25, email='alice@example.com', hobbies=[])
print(alice == Person("Alice", 25, "alice@example.com"))  # True
```

---

## 5. 异步编程基础

### 5.1 基本概念

```python
# 同步 vs 异步
# 同步：一个任务完成后再做下一个
# 异步：任务可以并发执行，等待时做其他事

import asyncio

# 定义异步函数
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # 异步等待，不阻塞
    print("World")

# 运行
asyncio.run(say_hello())
```

### 5.2 并发执行多个任务

```python
import asyncio
import time

async def fetch_data(name, delay):
    print(f"Start fetching {name}")
    await asyncio.sleep(delay)  # 模拟 IO 操作
    print(f"Done fetching {name}")
    return f"{name} data"

async def main():
    start = time.time()

    # 并发执行多个任务
    results = await asyncio.gather(
        fetch_data("A", 2),
        fetch_data("B", 1),
        fetch_data("C", 3)
    )

    print(f"Results: {results}")
    print(f"Total time: {time.time() - start:.2f}s")  # 约 3 秒，而非 6 秒

asyncio.run(main())
# 输出:
# Start fetching A
# Start fetching B
# Start fetching C
# Done fetching B
# Done fetching A
# Done fetching C
# Results: ['A data', 'B data', 'C data']
# Total time: 3.00s
```

### 5.3 实际应用场景

```python
import asyncio
import aiohttp  # 需要 pip install aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# 使用
# urls = ["http://example.com", "http://example.org"]
# results = asyncio.run(fetch_all_urls(urls))
```

---

## 6. 类型注解

### 6.1 基本类型注解

```python
# 变量注解
name: str = "Alice"
age: int = 25
height: float = 1.75
is_student: bool = True

# 函数注解
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

# None 类型
def say_hello() -> None:
    print("Hello!")
```

### 6.2 复杂类型

```python
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# 容器类型
names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 90, "Bob": 85}
point: Tuple[int, int] = (3, 4)
unique_ids: Set[int] = {1, 2, 3}

# Optional: 可能为 None
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # 可能返回 None

# Union: 多种类型之一
def process(value: Union[int, str]) -> str:
    return str(value)

# Any: 任意类型
def log(message: Any) -> None:
    print(message)
```

### 6.3 Python 3.10+ 新语法

```python
# 使用 | 替代 Union
def process(value: int | str) -> str:
    return str(value)

# 直接使用内置类型，不需要导入
def get_names() -> list[str]:
    return ["Alice", "Bob"]

def get_scores() -> dict[str, int]:
    return {"Alice": 90, "Bob": 85}
```

### 6.4 类型检查工具

```python
# 类型注解不会在运行时检查，需要使用工具

# 1. mypy: 静态类型检查
# pip install mypy
# mypy your_script.py

# 示例
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # mypy 会报错，运行时不会

# 2. IDE 支持
# VS Code 和 PyCharm 会根据类型注解提供更好的代码补全和错误提示
```

---

## 7. 常用内置函数

### 7.1 map, filter, reduce

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# map: 对每个元素应用函数
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# filter: 过滤元素
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# reduce: 累积计算
total = reduce(lambda a, b: a + b, numbers)
print(total)  # 15

# 但通常推荐用列表推导式或内置函数
squared = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
total = sum(numbers)
```

### 7.2 zip 和 enumerate

```python
# zip: 并行遍历多个可迭代对象
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age}")

# 创建字典
person_dict = dict(zip(names, ages))
print(person_dict)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# enumerate: 获取索引和值
for i, name in enumerate(names):
    print(f"{i}: {name}")

for i, name in enumerate(names, start=1):  # 从 1 开始
    print(f"{i}: {name}")
```

### 7.3 sorted 和 sort

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# sorted: 返回新列表
sorted_nums = sorted(numbers)
print(sorted_nums)  # [1, 1, 2, 3, 4, 5, 6, 9]
print(numbers)      # [3, 1, 4, 1, 5, 9, 2, 6] 原列表不变

# 降序
sorted_desc = sorted(numbers, reverse=True)
print(sorted_desc)  # [9, 6, 5, 4, 3, 2, 1, 1]

# 自定义排序
words = ["banana", "apple", "Cherry", "date"]
sorted_words = sorted(words, key=str.lower)  # 忽略大小写
print(sorted_words)  # ['apple', 'banana', 'Cherry', 'date']

# 复杂对象排序
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]
sorted_students = sorted(students, key=lambda x: x["score"], reverse=True)
print(sorted_students)

# sort: 原地排序
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]
```

### 7.4 其他实用函数

```python
# all / any
print(all([True, True, False]))  # False
print(any([True, False, False])) # True

# 检查列表中是否所有元素都满足条件
numbers = [2, 4, 6, 8]
print(all(x % 2 == 0 for x in numbers))  # True

# min / max
print(min([3, 1, 4]))  # 1
print(max([3, 1, 4]))  # 4

# 自定义 key
words = ["apple", "banana", "cherry"]
print(max(words, key=len))  # banana

# abs / round
print(abs(-5))       # 5
print(round(3.7))    # 4
print(round(3.14159, 2))  # 3.14

# isinstance
print(isinstance(5, int))        # True
print(isinstance("hi", str))     # True
print(isinstance([1, 2], list))  # True
```

---

## 8. 练习题

### 基础练习

1. 用列表推导式生成 1-100 中所有能被 3 整除但不能被 5 整除的数
2. 写一个装饰器，打印函数的执行时间
3. 写一个生成器，生成无限的素数序列
4. 定义一个 `Rectangle` 类，包含计算面积和周长的方法

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 列表推导式
result = [x for x in range(1, 101) if x % 3 == 0 and x % 5 != 0]
print(result)

# 2. 计时装饰器
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.5)
    return "Done"

slow_function()

# 3. 素数生成器
def prime_generator():
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1

primes = prime_generator()
for _ in range(10):
    print(next(primes), end=" ")  # 2 3 5 7 11 13 17 19 23 29

# 4. Rectangle 类
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"Rectangle({self.width} x {self.height})"

rect = Rectangle(5, 3)
print(rect)  # Rectangle(5 x 3)
print(f"Area: {rect.area()}")  # Area: 15
print(f"Perimeter: {rect.perimeter()}")  # Perimeter: 16
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [05-NumPy数组运算.md](./05-NumPy数组运算.md)

