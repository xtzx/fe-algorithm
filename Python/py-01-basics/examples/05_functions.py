#!/usr/bin/env python3
"""
05_functions.py - 函数演示

演示：
- 函数定义
- 参数类型
- 返回值
- 作用域
- Lambda
"""

# =============================================================================
# 1. 基本函数
# =============================================================================

print("=== 基本函数 ===")


def greet(name):
    """向用户打招呼（这是 docstring）"""
    return f"Hello, {name}!"


print(greet("Alice"))
print(greet.__doc__)  # 访问文档字符串


# 无返回值
def say_hello():
    print("Hello!")


result = say_hello()
print(f"无返回值函数返回: {result}")  # None

# =============================================================================
# 2. 参数类型
# =============================================================================

print("\n=== 参数类型 ===")


# 默认参数
def greet_with_default(name, greeting="Hello"):
    return f"{greeting}, {name}!"


print(greet_with_default("Bob"))
print(greet_with_default("Bob", "Hi"))


# 关键字参数
def create_user(name, age, city):
    return {"name": name, "age": age, "city": city}


user = create_user(name="Alice", age=25, city="NYC")
print(f"用户: {user}")

# 可以改变顺序
user = create_user(city="LA", name="Bob", age=30)
print(f"用户: {user}")


# *args：可变位置参数
def sum_all(*args):
    print(f"  args = {args}")
    return sum(args)


print(f"\nsum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")


# **kwargs：可变关键字参数
def print_info(**kwargs):
    print("  kwargs:")
    for key, value in kwargs.items():
        print(f"    {key} = {value}")


print("\nprint_info(name='Alice', age=25):")
print_info(name="Alice", age=25)


# 组合使用
def example(a, b, *args, **kwargs):
    print(f"  a = {a}, b = {b}")
    print(f"  args = {args}")
    print(f"  kwargs = {kwargs}")


print("\nexample(1, 2, 3, 4, x=5, y=6):")
example(1, 2, 3, 4, x=5, y=6)

# =============================================================================
# 3. 返回值
# =============================================================================

print("\n=== 返回值 ===")


# 多返回值（元组）
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)


nums = [1, 2, 3, 4, 5]
minimum, maximum, total = get_stats(nums)
print(f"最小: {minimum}, 最大: {maximum}, 总和: {total}")

# 也可以接收为元组
result = get_stats(nums)
print(f"作为元组: {result}")

# =============================================================================
# 4. 作用域（LEGB）
# =============================================================================

print("\n=== 作用域 ===")

x = "global"


def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(f"  inner 中 x = {x}")

    inner()
    print(f"  outer 中 x = {x}")


outer()
print(f"  全局 x = {x}")

# global 关键字
count = 0


def increment():
    global count
    count += 1


increment()
increment()
print(f"\n使用 global 后 count = {count}")


# nonlocal 关键字
def make_counter():
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter


counter = make_counter()
print(f"\n计数器: {counter()}, {counter()}, {counter()}")

# =============================================================================
# 5. Lambda 表达式
# =============================================================================

print("\n=== Lambda 表达式 ===")

# 基本用法
square = lambda x: x**2
add = lambda a, b: a + b

print(f"square(5) = {square(5)}")
print(f"add(2, 3) = {add(2, 3)}")

# 排序中使用
users = [
    {"name": "Charlie", "age": 35},
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
]

sorted_users = sorted(users, key=lambda u: u["age"])
print(f"\n按年龄排序: {[u['name'] for u in sorted_users]}")

sorted_users = sorted(users, key=lambda u: u["name"])
print(f"按姓名排序: {[u['name'] for u in sorted_users]}")

# =============================================================================
# 6. 函数作为一等公民
# =============================================================================

print("\n=== 函数作为一等公民 ===")


# 函数作为参数
def apply(func, value):
    return func(value)


result = apply(lambda x: x * 2, 5)
print(f"apply(double, 5) = {result}")


# 函数作为返回值
def make_multiplier(n):
    def multiplier(x):
        return x * n

    return multiplier


double = make_multiplier(2)
triple = make_multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")

# =============================================================================
# 7. 类型提示
# =============================================================================

print("\n=== 类型提示 ===")


def add_numbers(a: int, b: int) -> int:
    """带类型提示的加法函数"""
    return a + b


def greet_user(name: str) -> str:
    return f"Hello, {name}!"


print(add_numbers(1, 2))
print(greet_user("Alice"))

print("\n=== 运行完成 ===")

