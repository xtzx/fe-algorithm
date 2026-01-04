# 06. 函数

## 🎯 本节目标

- 掌握函数定义与调用
- 理解参数类型
- 学会多返回值
- 理解作用域规则

---

## 📝 函数定义

### 基本语法

```python
def greet(name):
    """向用户打招呼"""  # docstring
    return f"Hello, {name}!"

# 调用
message = greet("Alice")
print(message)  # Hello, Alice!
```

### JS 对照

```python
# Python
def add(a, b):
    return a + b
```

```javascript
// JS function
function add(a, b) {
    return a + b;
}

// JS arrow function
const add = (a, b) => a + b;
```

### 无返回值函数

```python
def print_hello():
    print("Hello!")
    # 没有 return，返回 None

result = print_hello()  # 打印 Hello!
print(result)  # None
```

---

## 📋 参数类型

### 1. 位置参数

```python
def greet(first_name, last_name):
    return f"Hello, {first_name} {last_name}!"

greet("John", "Doe")  # 按位置传递
```

### 2. 默认参数

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")           # Hello, Alice!
greet("Bob", "Hi")       # Hi, Bob!
```

**⚠️ 可变默认参数陷阱**

```python
# ❌ 错误：可变对象作为默认值
def add_item(item, items=[]):
    items.append(item)
    return items

add_item(1)  # [1]
add_item(2)  # [1, 2]  ← 意外！

# ✅ 正确：使用 None
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 3. 关键字参数

```python
def create_user(name, age, city):
    return {"name": name, "age": age, "city": city}

# 使用关键字参数
user = create_user(name="Alice", age=25, city="NYC")

# 可以改变顺序
user = create_user(city="NYC", name="Alice", age=25)

# 混合使用
user = create_user("Alice", city="NYC", age=25)
```

### 4. *args：可变位置参数

```python
def sum_all(*args):
    """接收任意数量的位置参数"""
    return sum(args)

sum_all(1, 2)        # 3
sum_all(1, 2, 3, 4)  # 10
```

### 5. **kwargs：可变关键字参数

```python
def print_info(**kwargs):
    """接收任意数量的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25)
# name: Alice
# age: 25
```

### 6. 组合使用

```python
def example(a, b, *args, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

example(1, 2, 3, 4, x=5, y=6)
# a=1, b=2
# args=(3, 4)
# kwargs={'x': 5, 'y': 6}
```

### 参数顺序规则

```python
def func(
    pos_only,      # 位置参数
    /,             # / 之前只能位置传递（Python 3.8+）
    standard,      # 标准参数（位置或关键字）
    *,             # * 之后只能关键字传递
    kw_only        # 关键字参数
):
    pass
```

---

## 📤 返回值

### 单返回值

```python
def square(x):
    return x ** 2
```

### 多返回值

```python
def get_stats(numbers):
    """返回多个值（实际是元组）"""
    return min(numbers), max(numbers), sum(numbers)

# 元组解包
minimum, maximum, total = get_stats([1, 2, 3, 4, 5])

# 也可以接收为元组
result = get_stats([1, 2, 3, 4, 5])
print(result)  # (1, 5, 15)
```

### 提前返回

```python
def validate_age(age):
    if age < 0:
        return False, "年龄不能为负"
    if age > 150:
        return False, "年龄不合理"
    return True, "验证通过"

valid, message = validate_age(25)
```

---

## 📖 文档字符串（Docstring）

```python
def calculate_area(width, height):
    """
    计算矩形面积。

    Args:
        width: 矩形宽度
        height: 矩形高度

    Returns:
        矩形的面积

    Raises:
        ValueError: 如果宽度或高度为负数

    Examples:
        >>> calculate_area(3, 4)
        12
    """
    if width < 0 or height < 0:
        raise ValueError("尺寸不能为负")
    return width * height

# 访问 docstring
print(calculate_area.__doc__)
help(calculate_area)
```

---

## 🌐 作用域（LEGB 规则）

Python 按 **L → E → G → B** 顺序查找变量：

1. **L**ocal：函数内部
2. **E**nclosing：外层函数
3. **G**lobal：模块级别
4. **B**uilt-in：内置

```python
x = "global"  # Global

def outer():
    x = "enclosing"  # Enclosing

    def inner():
        x = "local"  # Local
        print(x)

    inner()

outer()  # 输出: local
```

### global 关键字

```python
count = 0

def increment():
    global count  # 声明使用全局变量
    count += 1

increment()
print(count)  # 1
```

### nonlocal 关键字

```python
def outer():
    x = 0

    def inner():
        nonlocal x  # 声明使用外层变量
        x += 1

    inner()
    return x

print(outer())  # 1
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `global x` | 无需声明（直接访问） |
| `nonlocal x` | 闭包自动捕获 |

---

## 🔧 Lambda 表达式

```python
# 语法：lambda 参数: 表达式
square = lambda x: x ** 2
add = lambda a, b: a + b

print(square(5))  # 25
print(add(2, 3))  # 5

# 常用于排序
users = [{"name": "Bob", "age": 30}, {"name": "Alice", "age": 25}]
sorted_users = sorted(users, key=lambda u: u["age"])
```

### JS 对照

```javascript
// Python: lambda x: x ** 2
// JS:     x => x ** 2

// Python: lambda a, b: a + b
// JS:     (a, b) => a + b
```

---

## 🎭 函数作为一等公民

```python
# 函数赋值给变量
def greet(name):
    return f"Hello, {name}!"

say_hello = greet
print(say_hello("World"))

# 函数作为参数
def apply(func, value):
    return func(value)

result = apply(lambda x: x * 2, 5)  # 10

# 函数作为返回值
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
print(double(5))  # 10
```

---

## 📝 类型提示（Type Hints）

```python
def greet(name: str) -> str:
    """带类型提示的函数"""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# 可选参数
from typing import Optional

def find_user(user_id: int) -> Optional[dict]:
    """可能返回 None"""
    return None
```

---

## ✅ 本节要点

1. `def` 定义函数（无大括号）
2. 支持默认参数、关键字参数
3. `*args` 收集位置参数，`**kwargs` 收集关键字参数
4. 多返回值实际是元组
5. 作用域遵循 LEGB 规则
6. `global` 和 `nonlocal` 声明外部变量
7. `lambda` 创建匿名函数
8. 类型提示增强可读性

