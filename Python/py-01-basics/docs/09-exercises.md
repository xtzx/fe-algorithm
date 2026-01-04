# 09. 练习题

> 25 道练习题，分为基础、进阶、挑战三个级别

---

## 🟢 基础题（10 道）

### 1. 变量交换

**题目**：不使用临时变量，交换两个变量的值。

<details>
<summary>答案</summary>

```python
a, b = 10, 20
a, b = b, a
print(a, b)  # 20, 10
```

**思路**：Python 支持元组解包，可以同时赋值。

</details>

---

### 2. 类型判断

**题目**：判断一个变量是整数还是浮点数还是字符串。

<details>
<summary>答案</summary>

```python
def check_type(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return "整数"
    elif isinstance(value, float):
        return "浮点数"
    elif isinstance(value, str):
        return "字符串"
    else:
        return "其他类型"

print(check_type(42))      # 整数
print(check_type(3.14))    # 浮点数
print(check_type("hello")) # 字符串
print(check_type(True))    # 其他类型（bool 是 int 的子类）
```

**思路**：使用 `isinstance()`，注意 `bool` 是 `int` 的子类。

</details>

---

### 3. 字符串反转

**题目**：反转一个字符串。

<details>
<summary>答案</summary>

```python
s = "Hello, Python!"
reversed_s = s[::-1]
print(reversed_s)  # !nohtyP ,olleH
```

**思路**：使用切片的负步长 `[::-1]`。

</details>

---

### 4. 统计字符

**题目**：统计字符串中某个字符出现的次数。

<details>
<summary>答案</summary>

```python
s = "hello world"
char = "l"

# 方法 1：内置方法
count = s.count(char)

# 方法 2：循环
count = 0
for c in s:
    if c == char:
        count += 1

print(count)  # 3
```

</details>

---

### 5. 偶数筛选

**题目**：从列表中筛选出所有偶数。

<details>
<summary>答案</summary>

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 方法 1：列表推导式
evens = [x for x in numbers if x % 2 == 0]

# 方法 2：filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(evens)  # [2, 4, 6, 8, 10]
```

</details>

---

### 6. 阶乘计算

**题目**：计算 n 的阶乘。

<details>
<summary>答案</summary>

```python
def factorial(n):
    if n < 0:
        raise ValueError("负数没有阶乘")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 或者使用递归
def factorial_recursive(n):
    if n < 0:
        raise ValueError("负数没有阶乘")
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

print(factorial(5))  # 120
```

</details>

---

### 7. FizzBuzz

**题目**：打印 1-100，遇到 3 的倍数打印 Fizz，5 的倍数打印 Buzz，同时是 3 和 5 的倍数打印 FizzBuzz。

<details>
<summary>答案</summary>

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

</details>

---

### 8. 最大值查找

**题目**：不使用 max() 函数，找出列表中的最大值。

<details>
<summary>答案</summary>

```python
def find_max(numbers):
    if not numbers:
        raise ValueError("列表不能为空")
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

print(find_max([3, 1, 4, 1, 5, 9, 2, 6]))  # 9
```

</details>

---

### 9. 回文判断

**题目**：判断一个字符串是否是回文（忽略大小写和空格）。

<details>
<summary>答案</summary>

```python
def is_palindrome(s):
    # 去掉空格，转小写
    s = s.replace(" ", "").lower()
    return s == s[::-1]

print(is_palindrome("A man a plan a canal Panama"))  # True
print(is_palindrome("hello"))  # False
```

</details>

---

### 10. 温度转换

**题目**：实现摄氏度和华氏度互转。

<details>
<summary>答案</summary>

```python
def celsius_to_fahrenheit(c):
    """摄氏度转华氏度"""
    return c * 9 / 5 + 32

def fahrenheit_to_celsius(f):
    """华氏度转摄氏度"""
    return (f - 32) * 5 / 9

print(celsius_to_fahrenheit(0))    # 32.0
print(celsius_to_fahrenheit(100))  # 212.0
print(fahrenheit_to_celsius(98.6)) # 37.0
```

</details>

---

## 🟡 进阶题（10 道）

### 11. 函数参数

**题目**：编写一个函数，接收任意数量的数字参数，返回它们的平均值。

<details>
<summary>答案</summary>

```python
def average(*args):
    if not args:
        return 0
    return sum(args) / len(args)

print(average(1, 2, 3, 4, 5))  # 3.0
print(average(10, 20))         # 15.0
```

</details>

---

### 12. 字典操作

**题目**：合并两个字典，如果有相同的键，值相加。

<details>
<summary>答案</summary>

```python
def merge_dicts(d1, d2):
    result = d1.copy()
    for key, value in d2.items():
        if key in result:
            result[key] += value
        else:
            result[key] = value
    return result

# Python 3.9+ 可以用 | 运算符，但不会相加
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
print(merge_dicts(d1, d2))  # {'a': 1, 'b': 5, 'c': 4}
```

</details>

---

### 13. 文件读取

**题目**：读取文件内容，统计每行的字符数。

<details>
<summary>答案</summary>

```python
def count_line_chars(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            # strip() 去掉换行符
            print(f"第 {i} 行: {len(line.strip())} 个字符")

# 测试
# count_line_chars("sample.txt")
```

</details>

---

### 14. 列表去重

**题目**：去除列表中的重复元素，保持原有顺序。

<details>
<summary>答案</summary>

```python
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Python 3.7+ dict 保持顺序
def remove_duplicates_v2(lst):
    return list(dict.fromkeys(lst))

print(remove_duplicates([1, 2, 2, 3, 1, 4]))  # [1, 2, 3, 4]
```

</details>

---

### 15. 嵌套字典访问

**题目**：安全地访问嵌套字典，不存在时返回默认值。

<details>
<summary>答案</summary>

```python
def get_nested(d, *keys, default=None):
    """安全获取嵌套字典的值"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

data = {
    "user": {
        "profile": {
            "name": "Alice"
        }
    }
}

print(get_nested(data, "user", "profile", "name"))  # Alice
print(get_nested(data, "user", "settings", "theme", default="dark"))  # dark
```

</details>

---

### 16. 词频统计

**题目**：统计字符串中每个单词出现的次数。

<details>
<summary>答案</summary>

```python
def word_frequency(text):
    words = text.lower().split()
    freq = {}
    for word in words:
        # 去除标点
        word = word.strip(".,!?;:")
        freq[word] = freq.get(word, 0) + 1
    return freq

# 或者使用 Counter
from collections import Counter

def word_frequency_v2(text):
    words = text.lower().split()
    words = [w.strip(".,!?;:") for w in words]
    return dict(Counter(words))

text = "Hello world. Hello Python. Python is great!"
print(word_frequency(text))
# {'hello': 2, 'world': 1, 'python': 2, 'is': 1, 'great': 1}
```

</details>

---

### 17. 递归目录

**题目**：递归列出目录下的所有文件。

<details>
<summary>答案</summary>

```python
from pathlib import Path

def list_files(directory):
    path = Path(directory)
    for item in path.iterdir():
        if item.is_file():
            print(item)
        elif item.is_dir():
            list_files(item)

# 更简洁的方式
def list_files_v2(directory):
    for path in Path(directory).rglob("*"):
        if path.is_file():
            print(path)
```

</details>

---

### 18. 闭包计数器

**题目**：使用闭包实现一个计数器。

<details>
<summary>答案</summary>

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

</details>

---

### 19. 日期处理

**题目**：计算两个日期之间相差多少天。

<details>
<summary>答案</summary>

```python
from datetime import datetime

def days_between(date1_str, date2_str, fmt="%Y-%m-%d"):
    date1 = datetime.strptime(date1_str, fmt)
    date2 = datetime.strptime(date2_str, fmt)
    delta = abs(date2 - date1)
    return delta.days

print(days_between("2024-01-01", "2024-12-31"))  # 365
```

</details>

---

### 20. JSON 处理

**题目**：读取 JSON 文件，修改某个字段后写回。

<details>
<summary>答案</summary>

```python
import json

def update_json_field(filename, key, value):
    # 读取
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 修改
    data[key] = value

    # 写回
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# update_json_field("config.json", "version", "2.0")
```

</details>

---

## 🔴 挑战题（5 道）

### 21. 斐波那契生成器

**题目**：使用生成器实现斐波那契数列。

<details>
<summary>答案</summary>

```python
def fibonacci(n):
    """生成前 n 个斐波那契数"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

print(list(fibonacci(10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

</details>

---

### 22. 装饰器：计时

**题目**：实现一个装饰器，打印函数执行时间。

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
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # slow_function 执行耗时: 1.00xx 秒
```

</details>

---

### 23. 类实现：栈

**题目**：使用类实现一个栈（LIFO）。

<details>
<summary>答案</summary>

```python
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("栈为空")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("栈为空")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

    def __len__(self):
        return self.size()

stack = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())  # 2
print(stack.peek()) # 1
```

</details>

---

### 24. 命令行参数

**题目**：解析命令行参数，实现简单的计算器。

<details>
<summary>答案</summary>

```python
import sys

def calculator():
    if len(sys.argv) != 4:
        print("用法: python calc.py <num1> <op> <num2>")
        print("示例: python calc.py 10 + 5")
        sys.exit(1)

    try:
        num1 = float(sys.argv[1])
        op = sys.argv[2]
        num2 = float(sys.argv[3])
    except ValueError:
        print("错误: 参数必须是数字")
        sys.exit(1)

    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b != 0 else "错误: 除以零",
    }

    if op not in operations:
        print(f"不支持的运算符: {op}")
        sys.exit(1)

    result = operations[op](num1, num2)
    print(f"{num1} {op} {num2} = {result}")

if __name__ == "__main__":
    calculator()
```

</details>

---

### 25. 正则表达式

**题目**：使用正则表达式提取文本中的所有邮箱地址。

<details>
<summary>答案</summary>

```python
import re

def extract_emails(text):
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

text = """
联系我们：
support@example.com
admin@company.org
user.name+tag@domain.co.uk
"""

print(extract_emails(text))
# ['support@example.com', 'admin@company.org', 'user.name+tag@domain.co.uk']
```

</details>

