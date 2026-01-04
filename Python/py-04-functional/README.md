# P04: 函数式与装饰器

> 面向 JS/TS 资深工程师的 Python 函数式编程教程

## 🎯 学完后能做

- ✅ 编写和理解装饰器
- ✅ 使用生成器处理大数据
- ✅ 运用函数式编程思想

---

## 🚀 快速开始

```bash
cd examples
python3 01_higher_order_functions.py
```

---

## 📚 目录结构

```
py-04-functional/
├── README.md
├── docs/
│   ├── 01-higher-order-functions.md  # 高阶函数
│   ├── 02-lambda.md                  # lambda 表达式
│   ├── 03-closure.md                 # 闭包
│   ├── 04-decorators.md              # 装饰器
│   ├── 05-generators.md              # 生成器
│   ├── 06-iterators.md               # 迭代器
│   ├── 07-functools.md               # functools 模块
│   ├── 08-itertools.md               # itertools 模块
│   ├── 09-exercises.md               # 练习题
│   └── 10-interview-questions.md     # 面试题
├── examples/
├── exercises/
├── project/
│   └── decorator_lib/
└── scripts/
```

---

## ⚡ Python 函数式 vs JavaScript

| 特性 | Python | JavaScript |
|------|--------|------------|
| 匿名函数 | `lambda x: x*2` | `x => x*2` |
| 高阶函数 | `map`, `filter`, `reduce` | `map`, `filter`, `reduce` |
| 装饰器 | `@decorator` | 无原生支持 |
| 生成器 | `yield` | `function*` / `yield` |
| 闭包 | ✅ 支持 | ✅ 支持 |
| 惰性求值 | 生成器表达式 | 生成器函数 |

---

## 🔥 核心概念速查

### 高阶函数

```python
# 函数作为参数
def apply(func, x):
    return func(x)

apply(lambda x: x**2, 5)  # 25

# map/filter/reduce
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

### 装饰器

```python
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

### 生成器

```python
# 生成器函数
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 生成器表达式
squares = (x**2 for x in range(10))
```

### 闭包

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

---

## ⚠️ 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| **装饰器丢失元信息** | `func.__name__` 变成 `wrapper` | 使用 `@wraps(func)` |
| **闭包变量绑定** | 循环中的 lambda 捕获最后一个值 | 使用默认参数或生成器 |
| **生成器只能迭代一次** | 第二次迭代为空 | 重新创建生成器 |
| **groupby 未排序** | 结果不正确 | 先排序再分组 |
| **lru_cache 可变参数** | 缓存失效 | 使用不可变参数 |

---

## 📖 学习路径

1. [高阶函数](docs/01-higher-order-functions.md)
2. [lambda 表达式](docs/02-lambda.md)
3. [闭包](docs/03-closure.md)
4. [装饰器](docs/04-decorators.md)
5. [生成器](docs/05-generators.md)
6. [迭代器](docs/06-iterators.md)
7. [functools](docs/07-functools.md)
8. [itertools](docs/08-itertools.md)
9. [练习题](docs/09-exercises.md)
10. [面试题](docs/10-interview-questions.md)

---

## 🛠️ 小项目：实用装饰器库

```bash
python3 project/decorator_lib/main.py
```

实现 `@timer`、`@retry`、`@cache`、`@validate` 装饰器。

