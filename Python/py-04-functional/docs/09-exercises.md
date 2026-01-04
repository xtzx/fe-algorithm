# 09. 练习题

> 25 道练习题，覆盖函数式编程核心概念

---

## 📝 高阶函数（5 道）

### 1. 实现 compose 函数

**题目**：实现函数组合，`compose(f, g)(x)` 等价于 `f(g(x))`。

<details>
<summary>答案</summary>

```python
def compose(*funcs):
    def composed(x):
        for func in reversed(funcs):
            x = func(x)
        return x
    return composed

# 测试
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x**2

transform = compose(square, double, add_one)
print(transform(5))  # ((5+1)*2)**2 = 144
```

</details>

---

### 2. 实现 pipe 函数

**题目**：实现管道函数，从左到右执行。

<details>
<summary>答案</summary>

```python
def pipe(*funcs):
    def piped(x):
        for func in funcs:
            x = func(x)
        return x
    return piped

# 测试
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x**2

transform = pipe(add_one, double, square)
print(transform(5))  # ((5+1)*2)**2 = 144
```

</details>

---

### 3. 实现 curry 函数

**题目**：实现柯里化，将多参数函数转为单参数函数链。

<details>
<summary>答案</summary>

```python
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

# 测试
@curry
def add(x, y, z):
    return x + y + z

add_5 = add(5)
add_5_10 = add_5(10)
print(add_5_10(15))  # 30
```

</details>

---

### 4. 实现 memoize 函数

**题目**：实现记忆化装饰器，缓存函数结果。

<details>
<summary>答案</summary>

```python
from functools import wraps

def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# 测试
@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 快速返回
```

</details>

---

### 5. 实现 debounce 函数

**题目**：实现防抖函数，延迟执行。

<details>
<summary>答案</summary>

```python
import time
from functools import wraps

def debounce(delay):
    def decorator(func):
        last_call = [0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_call[0] >= delay:
                last_call[0] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

# 测试
@debounce(1)
def expensive_operation():
    print("执行操作")

expensive_operation()
expensive_operation()  # 1秒内不会执行
```

</details>

---

## 🎨 装饰器（8 道）

### 6. 实现计时装饰器

**题目**：实现 `@timer` 装饰器，记录函数执行时间。

<details>
<summary>答案</summary>

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

slow_function()
```

</details>

---

### 7. 实现重试装饰器

**题目**：实现 `@retry` 装饰器，失败时自动重试。

<details>
<summary>答案</summary>

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
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

</details>

---

### 8. 实现类型检查装饰器

**题目**：实现 `@validate_types` 装饰器，检查参数类型。

<details>
<summary>答案</summary>

```python
from functools import wraps
import inspect

def validate_types(**types):
    def decorator(func):
        sig = inspect.signature(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name, value in bound.arguments.items():
                if name in types and not isinstance(value, types[name]):
                    raise TypeError(f"{name} 应该是 {types[name].__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}
```

</details>

---

### 9. 实现日志装饰器

**题目**：实现 `@log_call` 装饰器，记录函数调用。

<details>
<summary>答案</summary>

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_call(func):
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

</details>

---

### 10. 实现单例装饰器

**题目**：实现 `@singleton` 装饰器，确保类只有一个实例。

<details>
<summary>答案</summary>

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("初始化数据库")

db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

</details>

---

### 11. 实现限流装饰器

**题目**：实现 `@rate_limit` 装饰器，限制函数调用频率。

<details>
<summary>答案</summary>

```python
import time
from functools import wraps
from collections import defaultdict

def rate_limit(max_calls=5, period=60):
    def decorator(func):
        calls = defaultdict(list)
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = id(args)
            calls[key] = [t for t in calls[key] if now - t < period]
            if len(calls[key]) >= max_calls:
                raise Exception(f"超过限流：{max_calls}次/{period}秒")
            calls[key].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=3, period=10)
def api_call():
    return "API 响应"
```

</details>

---

### 12. 实现缓存装饰器

**题目**：实现带 TTL（生存时间）的缓存装饰器。

<details>
<summary>答案</summary>

```python
import time
from functools import wraps

def cache_with_ttl(ttl=60):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                value, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return value
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator

@cache_with_ttl(ttl=10)
def expensive_function(n):
    print(f"计算 {n}...")
    return n ** 2
```

</details>

---

### 13. 实现权限检查装饰器

**题目**：实现 `@require_role` 装饰器，检查用户权限。

<details>
<summary>答案</summary>

```python
from functools import wraps

def require_role(*allowed_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = args[0] if args else kwargs.get("user")
            if not user or user.get("role") not in allowed_roles:
                raise PermissionError(f"需要角色: {allowed_roles}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_role("admin", "moderator")
def delete_post(user, post_id):
    return f"删除帖子 {post_id}"
```

</details>

---

## 🔄 生成器（7 道）

### 14. 实现斐波那契生成器

**题目**：实现无限斐波那契生成器。

<details>
<summary>答案</summary>

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
for i, n in enumerate(fib):
    if i >= 10:
        break
    print(n, end=" ")
# 0 1 1 2 3 5 8 13 21 34
```

</details>

---

### 15. 实现素数生成器

**题目**：实现无限素数生成器。

<details>
<summary>答案</summary>

```python
def primes():
    yield 2
    primes_list = [2]
    n = 3
    while True:
        if all(n % p != 0 for p in primes_list):
            primes_list.append(n)
            yield n
        n += 2

prime_gen = primes()
first_10 = [next(prime_gen) for _ in range(10)]
print(first_10)
```

</details>

---

### 16. 实现文件读取生成器

**题目**：实现逐行读取大文件的生成器。

<details>
<summary>答案</summary>

```python
def read_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# 使用
for line in read_lines("large_file.txt"):
    if "error" in line:
        print(line)
```

</details>

---

### 17. 实现分页生成器

**题目**：实现分页生成器，每次返回一页数据。

<details>
<summary>答案</summary>

```python
def paginate(items, page_size=10):
    for i in range(0, len(items), page_size):
        yield items[i:i + page_size]

data = list(range(100))
for page in paginate(data, page_size=20):
    print(f"处理页面: {len(page)} 条数据")
```

</details>

---

### 18. 实现滑动窗口生成器

**题目**：实现滑动窗口生成器。

<details>
<summary>答案</summary>

```python
from itertools import islice

def sliding_window(iterable, n):
    it = iter(iterable)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for x in it:
        window = window[1:] + (x,)
        yield window

numbers = [1, 2, 3, 4, 5, 6]
for window in sliding_window(numbers, 3):
    print(window)
```

</details>

---

### 19. 实现展平生成器

**题目**：实现展平嵌套列表的生成器。

<details>
<summary>答案</summary>

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3], [4, [5, 6]], 7]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7]
```

</details>

---

### 20. 实现批次生成器

**题目**：实现分批处理数据的生成器。

<details>
<summary>答案</summary>

```python
from itertools import islice

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

numbers = range(10)
for batch in batched(numbers, 3):
    print(batch)
```

</details>

---

## 🔧 itertools（5 道）

### 21. 实现所有组合

**题目**：使用 itertools 找出所有满足条件的组合。

<details>
<summary>答案</summary>

```python
from itertools import combinations

def find_combinations(items, target_sum):
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            if sum(combo) == target_sum:
                yield combo

numbers = [1, 2, 3, 4, 5]
for combo in find_combinations(numbers, 5):
    print(combo)
```

</details>

---

### 22. 实现排列生成器

**题目**：生成所有可能的排列。

<details>
<summary>答案</summary>

```python
from itertools import permutations

def all_permutations(items, length=None):
    if length is None:
        length = len(items)
    return permutations(items, length)

items = ["A", "B", "C"]
for perm in all_permutations(items, 2):
    print(perm)
```

</details>

---

### 23. 实现分组统计

**题目**：使用 groupby 按条件分组并统计。

<details>
<summary>答案</summary>

```python
from itertools import groupby

def group_by_length(words):
    sorted_words = sorted(words, key=len)
    for length, group in groupby(sorted_words, key=len):
        yield length, list(group)

words = ["apple", "pie", "banana", "cat", "dog"]
for length, group in group_by_length(words):
    print(f"长度 {length}: {group}")
```

</details>

---

### 24. 实现笛卡尔积生成器

**题目**：生成多个集合的笛卡尔积。

<details>
<summary>答案</summary>

```python
from itertools import product

def cartesian_product(*iterables):
    return product(*iterables)

colors = ["red", "blue"]
sizes = ["S", "M", "L"]
for combo in cartesian_product(colors, sizes):
    print(combo)
```

</details>

---

### 25. 实现链式迭代器

**题目**：使用 chain 连接多个可迭代对象。

<details>
<summary>答案</summary>

```python
from itertools import chain

def chain_iterables(*iterables):
    return chain.from_iterable(iterables)

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

for value in chain_iterables(list1, list2, list3):
    print(value)
```

</details>

---

## ✅ 练习建议

1. 先理解概念，再动手实现
2. 测试边界情况
3. 考虑性能和内存
4. 阅读标准库实现（如 functools.lru_cache）
5. 尝试优化和改进

