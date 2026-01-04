# 01. 高阶函数

## 🎯 本节目标

- 理解函数作为一等公民
- 掌握函数作为参数和返回值
- 熟练使用 map、filter、reduce
- 理解 sorted 的 key 参数

---

## 📝 什么是高阶函数

高阶函数（Higher-Order Function）是指：
1. **接受函数作为参数**
2. **返回函数作为结果**

的函数。

### 函数是一等公民

在 Python 中，函数和变量一样，可以：
- 赋值给变量
- 作为参数传递
- 作为返回值返回
- 存储在数据结构中

```python
# 函数可以赋值给变量
def greet(name):
    return f"Hello, {name}"

say_hello = greet
print(say_hello("Alice"))  # Hello, Alice

# 函数可以作为参数
def apply(func, value):
    return func(value)

result = apply(lambda x: x**2, 5)
print(result)  # 25

# 函数可以作为返回值
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
print(double(5))  # 10
```

### JS 对照

```javascript
// JavaScript 中函数也是一等公民
const greet = (name) => `Hello, ${name}`;
const sayHello = greet;

const apply = (func, value) => func(value);
apply(x => x**2, 5);  // 25

const makeMultiplier = (n) => (x) => x * n;
const double = makeMultiplier(2);
double(5);  // 10
```

---

## 🗺️ map

对每个元素应用函数，返回新迭代器。

```python
numbers = [1, 2, 3, 4, 5]

# 基本用法
squares = map(lambda x: x**2, numbers)
print(list(squares))  # [1, 4, 9, 16, 25]

# 使用普通函数
def double(x):
    return x * 2

doubled = list(map(double, numbers))
print(doubled)  # [2, 4, 6, 8, 10]

# 多个可迭代对象
a = [1, 2, 3]
b = [4, 5, 6]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # [5, 7, 9]
```

### map vs 列表推导式

```python
numbers = [1, 2, 3, 4, 5]

# map 方式
squares = list(map(lambda x: x**2, numbers))

# 推导式（更 Pythonic）
squares = [x**2 for x in numbers]
```

**何时用 map**：
- 已有现成函数时：`list(map(str, numbers))`
- 需要惰性求值时（不转 list）

---

## 🔍 filter

过滤元素，返回满足条件的迭代器。

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 基本用法
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [2, 4, 6, 8, 10]

# 使用普通函数
def is_even(x):
    return x % 2 == 0

evens = list(filter(is_even, numbers))

# 过滤 Falsy 值
data = [0, 1, "", "hello", None, [], [1, 2]]
clean = list(filter(None, data))
print(clean)  # [1, 'hello', [1, 2]]
```

### filter vs 列表推导式

```python
numbers = [1, 2, 3, 4, 5]

# filter 方式
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 推导式（更 Pythonic）
evens = [x for x in numbers if x % 2 == 0]
```

---

## 📊 reduce

累积操作，将序列归约为单个值。

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda acc, x: acc + x, numbers)
print(total)  # 15

# 等价于
total = sum(numbers)

# 求积
product = reduce(lambda acc, x: acc * x, numbers)
print(product)  # 120

# 带初始值
total = reduce(lambda acc, x: acc + x, numbers, 10)
print(total)  # 25

# 找最大值
max_val = reduce(lambda a, b: a if a > b else b, numbers)
print(max_val)  # 5
```

### JS 对照

```javascript
// JavaScript reduce
const numbers = [1, 2, 3, 4, 5];
const total = numbers.reduce((acc, x) => acc + x, 0);
const product = numbers.reduce((acc, x) => acc * x, 1);
```

---

## 📋 sorted 的 key 参数

`sorted()` 的 `key` 参数接受一个函数，用于提取排序键。

```python
# 按长度排序
words = ["apple", "pie", "banana", "cherry"]
sorted_words = sorted(words, key=len)
print(sorted_words)  # ['pie', 'apple', 'banana', 'cherry']

# 按绝对值排序
numbers = [-5, 2, -1, 4, -3]
sorted_nums = sorted(numbers, key=abs)
print(sorted_nums)  # [-1, 2, -3, 4, -5]

# 复杂对象排序
users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 20},
]

# 按年龄排序
by_age = sorted(users, key=lambda u: u["age"])
print([u["name"] for u in by_age])  # ['Charlie', 'Alice', 'Bob']

# 多字段排序
from operator import itemgetter
by_age_name = sorted(users, key=itemgetter("age", "name"))
```

### operator 模块

```python
from operator import itemgetter, attrgetter, methodcaller

# itemgetter：获取字典/元组的项
users = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
sorted(users, key=itemgetter("age"))

# attrgetter：获取对象属性
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 25), Person("Bob", 30)]
sorted(people, key=attrgetter("age"))

# methodcaller：调用方法
words = ["hello", "world", "python"]
sorted(words, key=methodcaller("upper"))
```

---

## 🔧 自定义高阶函数

```python
def compose(*funcs):
    """函数组合"""
    def composed(x):
        for func in reversed(funcs):
            x = func(x)
        return x
    return composed

# 使用
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x**2

# 组合：square(double(add_one(5)))
transform = compose(square, double, add_one)
print(transform(5))  # ((5+1)*2)**2 = 144

def pipe(*funcs):
    """管道（从左到右）"""
    def piped(x):
        for func in funcs:
            x = func(x)
        return x
    return piped

# 使用
transform = pipe(add_one, double, square)
print(transform(5))  # ((5+1)*2)**2 = 144
```

---

## 📊 JS 对照表

| Python | JavaScript | 说明 |
|--------|------------|------|
| `map(func, iterable)` | `array.map(func)` | 映射 |
| `filter(func, iterable)` | `array.filter(func)` | 过滤 |
| `reduce(func, iterable)` | `array.reduce(func)` | 累积 |
| `sorted(iterable, key=func)` | `array.sort((a,b) => ...)` | 排序 |
| `lambda x: x*2` | `x => x*2` | 匿名函数 |

---

## ✅ 本节要点

1. 函数是一等公民，可以赋值、传递、返回
2. `map` 映射，`filter` 过滤，`reduce` 累积
3. `sorted` 的 `key` 参数用于自定义排序
4. 推导式通常比 map/filter 更 Pythonic
5. `operator` 模块提供便捷的函数提取器

