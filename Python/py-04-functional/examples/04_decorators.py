#!/usr/bin/env python3
"""
04_decorators.py - 装饰器演示
"""

import time
from functools import wraps

# =============================================================================
# 1. 基础装饰器
# =============================================================================
print("=== 基础装饰器 ===")

def timer(func):
    """计时装饰器"""
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
    time.sleep(0.1)
    return "完成"

result = slow_function()
print(f"结果: {result}")

# =============================================================================
# 2. 带参数的装饰器
# =============================================================================
print("\n=== 带参数的装饰器 ===")

def retry(max_attempts=3, delay=1):
    """重试装饰器"""
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

@retry(max_attempts=3, delay=0.1)
def unstable_function():
    import random
    if random.random() < 0.5:
        raise ValueError("随机失败")
    return "成功"

try:
    result = unstable_function()
    print(f"结果: {result}")
except ValueError as e:
    print(f"最终失败: {e}")

# =============================================================================
# 3. 多个装饰器
# =============================================================================
print("\n=== 多个装饰器 ===")

def decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器1：前")
        result = func(*args, **kwargs)
        print("装饰器1：后")
        return result
    return wrapper

def decorator2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器2：前")
        result = func(*args, **kwargs)
        print("装饰器2：后")
        return result
    return wrapper

@decorator1
@decorator2
def my_function():
    print("函数执行")

my_function()

# =============================================================================
# 4. 类装饰器
# =============================================================================
print("\n=== 类装饰器 ===")

class Counter:
    """计数装饰器"""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 被调用了 {self.count} 次")
        return self.func(*args, **kwargs)

@Counter
def my_function():
    pass

my_function()
my_function()

# =============================================================================
# 5. 装饰类的装饰器
# =============================================================================
print("\n=== 装饰类的装饰器 ===")

def singleton(cls):
    """单例装饰器"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("初始化数据库连接")

db1 = Database()
db2 = Database()
print(f"db1 is db2: {db1 is db2}")

print("\n=== 运行完成 ===")

