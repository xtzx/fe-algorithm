# 10. 面试高频问题

> 10 个 Python 函数式编程面试高频问题

---

## 1. 什么是装饰器？手写一个计时装饰器？

<details>
<summary>参考答案</summary>

装饰器是修改或增强函数功能的语法糖，不改变函数定义。

**计时装饰器实现**：

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

**装饰器原理**：
```python
@timer
def func():
    pass

# 等价于
def func():
    pass
func = timer(func)
```

</details>

---

## 2. 装饰器的执行顺序是怎样的？

<details>
<summary>参考答案</summary>

多个装饰器**从下到上**执行，但执行顺序是**从上到下**。

```python
@decorator1
@decorator2
def my_function():
    pass

# 等价于
my_function = decorator1(decorator2(my_function))
```

**执行流程**：
1. `decorator2` 先包装函数
2. `decorator1` 再包装 `decorator2` 的结果
3. 调用时：`decorator1` → `decorator2` → `my_function`

**示例**：
```python
def d1(func):
    def wrapper():
        print("d1 前")
        func()
        print("d1 后")
    return wrapper

def d2(func):
    def wrapper():
        print("d2 前")
        func()
        print("d2 后")
    return wrapper

@d1
@d2
def f():
    print("函数")

f()
# d1 前
# d2 前
# 函数
# d2 后
# d1 后
```

</details>

---

## 3. 生成器和列表的区别？什么时候用生成器？

<details>
<summary>参考答案</summary>

| 特性 | 列表 | 生成器 |
|------|------|--------|
| 内存占用 | 高（所有值） | 低（一个值） |
| 创建速度 | 慢 | 快 |
| 访问速度 | 快（索引） | 慢（顺序） |
| 可重复迭代 | ✅ | ❌ |
| 长度 | 已知 | 未知 |

**何时用生成器**：
- ✅ 大数据处理（内存受限）
- ✅ 无限序列
- ✅ 管道处理
- ✅ 只需要遍历一次

**何时用列表**：
- ✅ 需要多次访问
- ✅ 需要索引访问
- ✅ 数据量小
- ✅ 需要长度信息

**示例**：
```python
# 列表：立即生成所有值
squares = [x**2 for x in range(1000000)]  # 占用大量内存

# 生成器：惰性生成
squares = (x**2 for x in range(1000000))  # 几乎不占内存
```

</details>

---

## 4. yield 和 return 的区别？

<details>
<summary>参考答案</summary>

| 特性 | return | yield |
|------|--------|-------|
| 返回值 | 立即返回 | 暂停并返回值 |
| 函数状态 | 结束 | 保持状态 |
| 调用次数 | 一次 | 多次 |
| 内存 | 一次性生成 | 惰性生成 |

**return**：
```python
def normal_function():
    return [1, 2, 3]  # 立即返回，函数结束

result = normal_function()  # [1, 2, 3]
```

**yield**：
```python
def generator_function():
    yield 1  # 返回 1，暂停
    yield 2  # 返回 2，暂停
    yield 3  # 返回 3，结束

gen = generator_function()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

**关键区别**：
- `return` 后函数结束，无法继续执行
- `yield` 后函数暂停，可以继续执行

</details>

---

## 5. lru_cache 的原理和使用场景？

<details>
<summary>参考答案</summary>

**LRU（Least Recently Used）**：最近最少使用缓存。

**原理**：
- 使用字典存储函数参数和结果的映射
- 维护访问顺序（双向链表）
- 缓存满时删除最久未使用的项

**使用场景**：
- ✅ 递归函数（如斐波那契）
- ✅ 计算密集型函数
- ✅ 相同参数频繁调用
- ❌ 可变参数（不能哈希）
- ❌ 结果经常变化

**示例**：
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 快速返回（有缓存）

# 查看缓存统计
print(fibonacci.cache_info())
# CacheInfo(hits=28, misses=31, maxsize=128, currsize=31)
```

**注意事项**：
- 参数必须是可哈希的
- 缓存会占用内存
- 不适合结果经常变化的函数

</details>

---

## 6. 如何实现一个带参数的装饰器？

<details>
<summary>参考答案</summary>

带参数的装饰器需要**三层嵌套函数**。

**结构**：
```python
def decorator_with_args(arg1, arg2):
    """装饰器工厂（外层）"""
    def decorator(func):
        """真正的装饰器（中层）"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """包装函数（内层）"""
            # 可以使用 arg1, arg2
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_with_args("param1", "param2")
def my_function():
    pass
```

**示例：重试装饰器**：
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
    # ...
    pass
```

</details>

---

## 7. 闭包是什么？Python 中如何创建闭包？

<details>
<summary>参考答案</summary>

**闭包**：内部函数引用了外部函数的变量，即使外部函数已返回，内部函数仍可访问这些变量。

**特征**：
1. 嵌套函数
2. 内部函数引用外部变量
3. 外部函数返回内部函数
4. 状态保持

**创建闭包**：
```python
def outer(x):
    # 外部变量
    def inner(y):
        # 引用外部变量 x
        return x + y
    return inner  # 返回内部函数

# 创建闭包
add_5 = outer(5)
print(add_5(10))  # 15（x=5 被"记住"了）
```

**修改外部变量**：
```python
def make_counter():
    count = 0

    def counter():
        nonlocal count  # ✅ 必须使用 nonlocal
        count += 1
        return count

    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

**常见用途**：
- 工厂函数
- 状态保持
- 延迟计算
- 配置管理

</details>

---

## 8. itertools.groupby 的使用注意事项？

<details>
<summary>参考答案</summary>

**⚠️ 重要**：`groupby` **要求数据已排序**！

**错误用法**：
```python
from itertools import groupby

data = [("A", 1), ("B", 2), ("A", 3)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1)]
# B [('B', 2)]
# A [('A', 3)]  # A 被分成两组！
```

**正确用法**：
```python
from itertools import groupby

data = [("A", 1), ("B", 2), ("A", 3)]
data_sorted = sorted(data, key=lambda x: x[0])  # ✅ 先排序

for key, group in groupby(data_sorted, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1), ('A', 3)]
# B [('B', 2)]
```

**原因**：
- `groupby` 只合并**连续的相同键**
- 未排序时，相同键可能不连续

**实际应用**：
```python
from itertools import groupby

words = ["apple", "pie", "banana", "cat", "dog"]
words_sorted = sorted(words, key=len)

for length, group in groupby(words_sorted, key=len):
    print(f"长度 {length}: {list(group)}")
```

</details>

---

## 9. 生成器如何处理大文件？

<details>
<summary>参考答案</summary>

生成器可以**逐行读取**大文件，不一次性加载到内存。

**方法 1：逐行读取**：
```python
def read_large_file(filename):
    """逐行读取大文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# 使用
for line in read_large_file("huge_file.txt"):
    if "error" in line:
        print(line)
```

**方法 2：分块读取**：
```python
def read_in_chunks(filename, chunk_size=1024):
    """分块读取"""
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# 使用
for chunk in read_in_chunks("large_file.bin", chunk_size=4096):
    process(chunk)
```

**优势**：
- ✅ 内存占用小（只加载当前行/块）
- ✅ 可以处理任意大小的文件
- ✅ 惰性求值，按需读取

**对比**：
```python
# ❌ 列表：一次性加载所有行
lines = open("huge_file.txt").readlines()  # 内存爆炸

# ✅ 生成器：逐行读取
for line in open("huge_file.txt"):
    process(line)  # 内存友好
```

</details>

---

## 10. 什么是惰性求值？

<details>
<summary>参考答案</summary>

**惰性求值（Lazy Evaluation）**：延迟计算，只在需要时才计算值。

**特点**：
- 不立即生成所有值
- 按需生成
- 节省内存
- 可以表示无限序列

**Python 中的惰性求值**：
1. **生成器表达式**：
```python
# 惰性：不立即生成
squares = (x**2 for x in range(1000000))

# 需要时才生成
for value in squares:
    if value > 100:
        break
```

2. **生成器函数**：
```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a  # 惰性生成
        a, b = b, a + b

# 只生成需要的部分
fib = fibonacci()
first_10 = [next(fib) for _ in range(10)]
```

3. **map/filter**：
```python
# 惰性迭代器
numbers = range(1000000)
squares = map(lambda x: x**2, numbers)  # 不立即计算

# 需要时才计算
first_5 = list(islice(squares, 5))
```

**优势**：
- ✅ 内存效率高
- ✅ 可以处理无限序列
- ✅ 按需计算，节省时间

**对比**：
```python
# 立即求值（Eager）
squares = [x**2 for x in range(1000000)]  # 立即生成所有值

# 惰性求值（Lazy）
squares = (x**2 for x in range(1000000))  # 不生成，需要时才生成
```

</details>

---

## ✅ 面试准备建议

1. **理解原理**：不只是记忆，要理解为什么
2. **手写代码**：能独立实现常用装饰器和生成器
3. **性能考虑**：理解内存和时间复杂度
4. **实际应用**：能说出使用场景
5. **常见陷阱**：了解并避免常见错误

