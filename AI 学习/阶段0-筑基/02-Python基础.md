# 🐍 02 - Python 基础

> 掌握 Python 核心语法，为 AI 学习打下基础

---

## 目录

1. [变量与数据类型](#1-变量与数据类型)
2. [运算符](#2-运算符)
3. [条件语句](#3-条件语句)
4. [循环语句](#4-循环语句)
5. [函数](#5-函数)
6. [数据结构](#6-数据结构)
7. [练习题](#7-练习题)

---

## 1. 变量与数据类型

### 1.1 变量赋值

```python
# Python 是动态类型语言，不需要声明类型
name = "Alice"      # 字符串
age = 25            # 整数
height = 1.75       # 浮点数
is_student = True   # 布尔值

# 同时赋值多个变量
x, y, z = 1, 2, 3
a = b = c = 0  # 都赋值为 0

# 查看类型
print(type(name))    # <class 'str'>
print(type(age))     # <class 'int'>
print(type(height))  # <class 'float'>
print(type(is_student))  # <class 'bool'>
```

### 1.2 基本数据类型

```python
# 整数（int）- 无大小限制
num1 = 42
num2 = 10**100  # 很大的数也可以

# 浮点数（float）
pi = 3.14159
scientific = 1.5e-10  # 科学计数法

# 字符串（str）
s1 = 'Hello'
s2 = "World"
s3 = '''多行
字符串'''
s4 = """也可以
用双引号"""

# 布尔值（bool）
flag1 = True
flag2 = False

# None（空值）
nothing = None
```

### 1.3 类型转换

```python
# 显式类型转换
x = "123"
y = int(x)      # 字符串转整数
z = float(x)    # 字符串转浮点数
s = str(456)    # 数字转字符串

# 转布尔值
bool(0)      # False
bool(1)      # True
bool("")     # False
bool("hi")   # True
bool([])     # False
bool([1])    # True

# 实用：字符串分割后转数字
nums = "1,2,3,4,5".split(",")
nums = [int(n) for n in nums]  # [1, 2, 3, 4, 5]
```

### 1.4 字符串操作

```python
s = "Hello, World!"

# 基本操作
print(len(s))           # 13 - 长度
print(s.lower())        # hello, world! - 小写
print(s.upper())        # HELLO, WORLD! - 大写
print(s.strip())        # 去除首尾空白
print(s.replace("World", "Python"))  # Hello, Python!

# 索引和切片
print(s[0])       # H - 第一个字符
print(s[-1])      # ! - 最后一个字符
print(s[0:5])     # Hello - 前5个
print(s[7:])      # World! - 从第7个开始
print(s[::-1])    # !dlroW ,olleH - 反转

# 分割和连接
words = s.split(", ")  # ['Hello', 'World!']
joined = "-".join(words)  # Hello-World!

# f-string 格式化（推荐）
name = "Alice"
age = 25
print(f"{name} is {age} years old")  # Alice is 25 years old
print(f"π ≈ {3.14159:.2f}")  # π ≈ 3.14

# 检查
print("Hello" in s)      # True
print(s.startswith("He"))  # True
print(s.endswith("!"))    # True
```

---

## 2. 运算符

### 2.1 算术运算符

```python
a, b = 10, 3

print(a + b)   # 13 - 加法
print(a - b)   # 7 - 减法
print(a * b)   # 30 - 乘法
print(a / b)   # 3.333... - 除法（返回浮点数）
print(a // b)  # 3 - 整除（向下取整）
print(a % b)   # 1 - 取余
print(a ** b)  # 1000 - 幂运算

# 复合赋值
x = 10
x += 5   # x = x + 5 = 15
x -= 3   # x = x - 3 = 12
x *= 2   # x = x * 2 = 24
x /= 4   # x = x / 4 = 6.0
```

### 2.2 比较运算符

```python
a, b = 10, 5

print(a == b)   # False - 等于
print(a != b)   # True - 不等于
print(a > b)    # True - 大于
print(a < b)    # False - 小于
print(a >= b)   # True - 大于等于
print(a <= b)   # False - 小于等于

# 链式比较
x = 5
print(1 < x < 10)  # True - 等价于 1 < x and x < 10
```

### 2.3 逻辑运算符

```python
a, b = True, False

print(a and b)  # False - 与
print(a or b)   # True - 或
print(not a)    # False - 非

# 短路求值
# and: 如果第一个为 False，不计算第二个
# or: 如果第一个为 True，不计算第二个
x = 5
result = x > 0 and x < 10  # True
result = x < 0 or x > 3    # True

# 实用：默认值
name = None
display_name = name or "Anonymous"  # "Anonymous"
```

### 2.4 成员运算符

```python
# in / not in
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# 字典中检查键
person = {"name": "Alice", "age": 25}
print("name" in person)  # True
print("height" in person)  # False
```

---

## 3. 条件语句

### 3.1 if-elif-else

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")  # Grade: B
```

### 3.2 三元表达式

```python
age = 20

# 传统写法
if age >= 18:
    status = "Adult"
else:
    status = "Minor"

# 三元表达式
status = "Adult" if age >= 18 else "Minor"
print(status)  # Adult
```

### 3.3 条件表达式的真假判断

```python
# 以下都被视为 False:
# - False
# - None
# - 0, 0.0
# - "", '', """"""
# - [], (), {}, set()

# 实用技巧
items = []
if items:  # 相当于 if len(items) > 0
    print("有数据")
else:
    print("没有数据")  # 输出这个

name = ""
if name:
    print(f"Hello, {name}")
else:
    print("Name is empty")  # 输出这个
```

---

## 4. 循环语句

### 4.1 for 循环

```python
# 遍历列表
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Python":
    print(char)

# range() 函数
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8):   # 2, 3, 4, 5, 6, 7
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (步长为2)
    print(i)

# enumerate() - 同时获取索引和值
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 输出:
# 0: apple
# 1: banana
# 2: cherry

# zip() - 同时遍历多个列表
names = ["Alice", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")
```

### 4.2 while 循环

```python
# 基本用法
count = 0
while count < 5:
    print(count)
    count += 1

# 无限循环 + break
while True:
    user_input = input("Enter 'q' to quit: ")
    if user_input == 'q':
        break
    print(f"You entered: {user_input}")
```

### 4.3 循环控制

```python
# break - 跳出循环
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# continue - 跳过本次，继续下一次
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # 1, 3, 5, 7, 9

# else 子句（循环正常结束时执行）
for i in range(5):
    if i == 10:
        break
else:
    print("Loop completed normally")  # 会执行
```

### 4.4 嵌套循环

```python
# 打印乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()

# 遍历二维列表
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for item in row:
        print(item, end=" ")
    print()
```

---

## 5. 函数

### 5.1 定义和调用

```python
# 基本函数
def greet(name):
    """向指定人打招呼"""  # 文档字符串
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Hello, Alice!

# 无返回值
def say_hello():
    print("Hello!")

say_hello()  # Hello!

# 返回多个值
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])
print(f"Min: {minimum}, Max: {maximum}, Avg: {average}")
```

### 5.2 参数类型

```python
# 位置参数
def add(a, b):
    return a + b

add(1, 2)  # 3

# 默认参数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")           # Hello, Alice!
greet("Bob", "Hi")       # Hi, Bob!

# 关键字参数
def create_profile(name, age, city="Unknown"):
    return {"name": name, "age": age, "city": city}

create_profile("Alice", 25)
create_profile(name="Bob", city="NYC", age=30)  # 顺序可以变

# *args - 可变位置参数
def sum_all(*numbers):
    return sum(numbers)

sum_all(1, 2, 3, 4, 5)  # 15

# **kwargs - 可变关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="NYC")
```

### 5.3 作用域

```python
# 全局变量 vs 局部变量
global_var = "I'm global"

def my_function():
    local_var = "I'm local"
    print(global_var)  # 可以读取全局变量
    print(local_var)

my_function()
# print(local_var)  # 错误！局部变量在函数外不可见

# 修改全局变量需要 global 关键字
counter = 0

def increment():
    global counter
    counter += 1

increment()
print(counter)  # 1
```

### 5.4 Lambda 表达式

```python
# 匿名函数
square = lambda x: x ** 2
print(square(5))  # 25

add = lambda a, b: a + b
print(add(3, 4))  # 7

# 常见用途：作为参数传递
numbers = [3, 1, 4, 1, 5, 9]
sorted_numbers = sorted(numbers, key=lambda x: -x)  # 降序
print(sorted_numbers)  # [9, 5, 4, 3, 1, 1]

# 配合 map, filter 使用
nums = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, nums))  # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
```

---

## 6. 数据结构

### 6.1 列表（List）

```python
# 创建
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# 访问
print(fruits[0])    # apple
print(fruits[-1])   # cherry
print(fruits[1:3])  # ['banana', 'cherry']

# 修改
fruits[0] = "avocado"
fruits.append("date")     # 末尾添加
fruits.insert(1, "berry")  # 指定位置插入
fruits.extend(["elderberry", "fig"])  # 扩展

# 删除
fruits.remove("banana")   # 删除指定元素
popped = fruits.pop()     # 删除并返回最后一个
del fruits[0]             # 删除指定索引

# 常用方法
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(len(numbers))         # 8
print(numbers.count(1))     # 2
print(numbers.index(4))     # 2
numbers.sort()              # 原地排序
numbers.reverse()           # 原地反转
print(sorted(numbers))      # 返回新列表，不修改原列表

# 列表推导式
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, ..., 81]
evens = [x for x in range(20) if x % 2 == 0]  # [0, 2, 4, ..., 18]
```

### 6.2 元组（Tuple）

```python
# 元组是不可变的列表
point = (3, 4)
colors = ("red", "green", "blue")

# 访问
print(point[0])  # 3
x, y = point     # 解包

# 不能修改
# point[0] = 5  # 错误！

# 单元素元组
single = (1,)  # 注意逗号

# 用途：函数返回多个值、字典的键
def get_coordinates():
    return (10, 20)

coords = get_coordinates()
print(coords)  # (10, 20)
```

### 6.3 字典（Dict）

```python
# 创建
person = {"name": "Alice", "age": 25, "city": "NYC"}
empty = {}
from_pairs = dict([("a", 1), ("b", 2)])

# 访问
print(person["name"])        # Alice
print(person.get("height"))  # None (不存在时返回 None)
print(person.get("height", 170))  # 170 (默认值)

# 修改和添加
person["age"] = 26           # 修改
person["email"] = "a@b.com"  # 添加新键

# 删除
del person["city"]
popped = person.pop("email", None)

# 遍历
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

for key in person.keys():
    print(key)

for value in person.values():
    print(value)

# 字典推导式
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 6.4 集合（Set）

```python
# 创建（无序、不重复）
fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# 添加和删除
fruits.add("date")
fruits.remove("banana")  # 不存在会报错
fruits.discard("xxx")    # 不存在不报错

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # {1, 2, 3, 4, 5, 6} - 并集
print(a & b)   # {3, 4} - 交集
print(a - b)   # {1, 2} - 差集
print(a ^ b)   # {1, 2, 5, 6} - 对称差集

# 用途：去重
names = ["Alice", "Bob", "Alice", "Charlie", "Bob"]
unique_names = list(set(names))  # ['Alice', 'Bob', 'Charlie']
```

---

## 7. 练习题

### 基础练习

1. 写一个函数，判断一个数是否为质数
2. 写一个函数，计算斐波那契数列的第 n 项
3. 写一个函数，反转一个字符串
4. 写一个函数，统计一个列表中每个元素出现的次数

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 判断质数
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(17))  # True
print(is_prime(18))  # False

# 2. 斐波那契数列
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print(fibonacci(10))  # 55

# 3. 反转字符串
def reverse_string(s):
    return s[::-1]

print(reverse_string("hello"))  # olleh

# 4. 统计元素出现次数
def count_elements(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts

print(count_elements([1, 2, 2, 3, 3, 3]))  # {1: 1, 2: 2, 3: 3}
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [03-Python进阶.md](./03-Python进阶.md)

