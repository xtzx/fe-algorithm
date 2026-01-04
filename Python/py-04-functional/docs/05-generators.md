# 05. 生成器（Generator）

## 🎯 本节目标

- 理解生成器的概念和优势
- 掌握 yield 关键字
- 使用生成器表达式
- 处理大数据场景

---

## 📝 什么是生成器

生成器（Generator）是一种**惰性求值**的迭代器，可以逐个产生值，而不是一次性生成所有值。

### 生成器函数

使用 `yield` 关键字的函数就是生成器函数。

```python
def countdown(n):
    """倒计时生成器"""
    while n > 0:
        yield n
        n -= 1

# 创建生成器对象
gen = countdown(5)
print(type(gen))  # <class 'generator'>

# 逐个获取值
print(next(gen))  # 5
print(next(gen))  # 4
print(next(gen))  # 3
```

### yield vs return

| 特性 | return | yield |
|------|--------|-------|
| 返回值 | 立即返回 | 暂停并返回值 |
| 函数状态 | 结束 | 保持状态 |
| 调用次数 | 一次 | 多次 |
| 内存 | 一次性生成 | 惰性生成 |

```python
# return：函数结束
def normal_function():
    return [1, 2, 3]  # 立即返回列表

# yield：函数暂停
def generator_function():
    yield 1  # 返回 1，暂停
    yield 2  # 返回 2，暂停
    yield 3  # 返回 3，结束
```

---

## 🔄 生成器的工作原理

### 执行流程

```python
def simple_generator():
    print("开始")
    yield 1
    print("中间")
    yield 2
    print("结束")

gen = simple_generator()
print("创建生成器")

print(next(gen))  # 开始 \n 1
print(next(gen))  # 中间 \n 2
# print(next(gen))  # StopIteration
```

### 状态保持

```python
def fibonacci():
    """斐波那契生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
for i in range(10):
    print(next(fib), end=" ")
# 0 1 1 2 3 5 8 13 21 34
```

---

## 📊 生成器表达式

类似列表推导式，但使用圆括号。

```python
# 列表推导式（立即生成）
squares_list = [x**2 for x in range(10)]
print(squares_list)  # [0, 1, 4, 9, 16, ...]

# 生成器表达式（惰性生成）
squares_gen = (x**2 for x in range(10))
print(squares_gen)  # <generator object <genexpr> at 0x...>
print(list(squares_gen))  # [0, 1, 4, 9, 16, ...]
```

### 优势：内存效率

```python
# ❌ 列表：占用大量内存
big_list = [x**2 for x in range(1000000)]  # 立即生成所有值

# ✅ 生成器：几乎不占内存
big_gen = (x**2 for x in range(1000000))  # 只生成需要的值

# 使用
for value in big_gen:
    if value > 100:
        break
    print(value)
```

---

## 🚀 惰性求值的优势

### 1. 内存效率

```python
# 处理大文件
def read_large_file(filename):
    """逐行读取大文件"""
    with open(filename) as f:
        for line in f:
            yield line.strip()

# 不需要一次性加载整个文件到内存
for line in read_large_file("huge_file.txt"):
    process(line)
```

### 2. 无限序列

```python
def natural_numbers():
    """自然数生成器"""
    n = 1
    while True:
        yield n
        n += 1

# 只生成需要的部分
nums = natural_numbers()
first_10 = [next(nums) for _ in range(10)]
print(first_10)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 3. 管道处理

```python
def numbers():
    for i in range(10):
        yield i

def squares(iterable):
    for x in iterable:
        yield x**2

def evens(iterable):
    for x in iterable:
        if x % 2 == 0:
            yield x

# 组合管道
result = list(evens(squares(numbers())))
print(result)  # [0, 4, 16, 36, 64]
```

---

## 🔗 yield from（委托生成器）

`yield from` 用于委托给另一个生成器。

### 基本用法

```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    yield from generator1()
    yield from generator2()

for value in combined():
    print(value)
# 1
# 2
# 3
# 4
```

### 展平嵌套结构

```python
def flatten(nested):
    """展平嵌套列表"""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3], [4, [5, 6]], 7]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7]
```

---

## 📤 send() 和 close()

### send()：向生成器发送值

```python
def accumulator():
    """累加器"""
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)  # 启动生成器（必须）

print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35
```

### close()：关闭生成器

```python
def countdown(n):
    try:
        while n > 0:
            yield n
            n -= 1
    except GeneratorExit:
        print("生成器被关闭")

gen = countdown(5)
print(next(gen))  # 5
gen.close()       # 生成器被关闭
```

---

## 🎯 实际应用场景

### 1. 读取大文件

```python
def read_lines(filename):
    """逐行读取"""
    with open(filename) as f:
        for line in f:
            yield line.strip()

# 处理大文件而不占内存
for line in read_lines("large_file.txt"):
    if "error" in line:
        print(line)
```

### 2. 分页处理

```python
def paginate(items, page_size=10):
    """分页生成器"""
    for i in range(0, len(items), page_size):
        yield items[i:i + page_size]

data = list(range(100))
for page in paginate(data, page_size=20):
    print(f"处理页面: {page[:5]}...")  # 只处理当前页
```

### 3. 数据流处理

```python
def filter_positive(numbers):
    """过滤正数"""
    for n in numbers:
        if n > 0:
            yield n

def square(numbers):
    """平方"""
    for n in numbers:
        yield n**2

# 组合处理
numbers = [-2, -1, 0, 1, 2, 3]
result = list(square(filter_positive(numbers)))
print(result)  # [1, 4, 9]
```

### 4. 无限序列

```python
def primes():
    """素数生成器"""
    yield 2
    primes_list = [2]
    n = 3
    while True:
        if all(n % p != 0 for p in primes_list):
            primes_list.append(n)
            yield n
        n += 2

# 获取前 10 个素数
prime_gen = primes()
first_10_primes = [next(prime_gen) for _ in range(10)]
print(first_10_primes)
```

---

## ⚠️ 常见陷阱

### 1. 生成器只能迭代一次

```python
gen = (x**2 for x in range(5))

print(list(gen))  # [0, 1, 4, 9, 16]
print(list(gen))  # []（已耗尽）

# ✅ 解决：重新创建
gen = (x**2 for x in range(5))
print(list(gen))  # [0, 1, 4, 9, 16]
```

### 2. 生成器表达式 vs 列表推导式

```python
# 列表推导式：立即求值
squares = [x**2 for x in range(10)]  # 已生成所有值

# 生成器表达式：惰性求值
squares = (x**2 for x in range(10))  # 还未生成值

# 需要时再转换
result = list(squares)
```

### 3. 在生成器中使用 return

```python
def generator_with_return():
    yield 1
    yield 2
    return "结束"  # 返回值会被忽略（Python 3.3+）

gen = generator_with_return()
for value in gen:
    print(value)
# 1
# 2
```

---

## 🆚 生成器 vs 列表

| 特性 | 列表 | 生成器 |
|------|------|--------|
| 内存占用 | 高（所有值） | 低（一个值） |
| 创建速度 | 慢 | 快 |
| 访问速度 | 快（索引） | 慢（顺序） |
| 可重复迭代 | ✅ | ❌ |
| 长度 | 已知 | 未知 |

**何时用生成器**：
- 大数据处理
- 无限序列
- 管道处理
- 内存受限

**何时用列表**：
- 需要多次访问
- 需要索引
- 数据量小

---

## ✅ 本节要点

1. 生成器使用 `yield` 关键字
2. 生成器是惰性求值，节省内存
3. 生成器表达式：`(x for x in range(10))`
4. `yield from` 委托给其他生成器
5. `send()` 向生成器发送值
6. 生成器只能迭代一次
7. 适合处理大数据和无限序列

