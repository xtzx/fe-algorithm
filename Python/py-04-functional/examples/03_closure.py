#!/usr/bin/env python3
"""
03_closure.py - 闭包演示
"""

# =============================================================================
# 1. 基本闭包
# =============================================================================
print("=== 基本闭包 ===")

def outer(x):
    """外部函数"""
    def inner(y):
        """内部函数引用外部变量 x"""
        return x + y
    return inner

add_5 = outer(5)
print(f"add_5(10) = {add_5(10)}")

add_3 = outer(3)
print(f"add_3(10) = {add_3(10)}")

# =============================================================================
# 2. 状态保持
# =============================================================================
print("\n=== 状态保持 ===")

def make_counter():
    """创建计数器"""
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter

c1 = make_counter()
c2 = make_counter()

print("c1:", c1(), c1(), c1())
print("c2:", c2(), c2())

# =============================================================================
# 3. nonlocal 关键字
# =============================================================================
print("\n=== nonlocal 关键字 ===")

def make_counter_without_nonlocal():
    """不使用 nonlocal（会出错）"""
    count = 0

    def counter():
        # count += 1  # ❌ UnboundLocalError
        return count

    return counter

def make_counter_with_nonlocal():
    """使用 nonlocal"""
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter

counter = make_counter_with_nonlocal()
print(f"counter() = {counter()}")
print(f"counter() = {counter()}")

# =============================================================================
# 4. 工厂函数
# =============================================================================
print("\n=== 工厂函数 ===")

def make_logger(prefix):
    """创建日志记录器"""
    def log(message):
        print(f"[{prefix}] {message}")
    return log

info_logger = make_logger("INFO")
error_logger = make_logger("ERROR")

info_logger("系统启动")
error_logger("发生错误")

# =============================================================================
# 5. 配置函数
# =============================================================================
print("\n=== 配置函数 ===")

def make_multiplier(n):
    """创建乘法器"""
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")

# =============================================================================
# 6. 延迟计算
# =============================================================================
print("\n=== 延迟计算 ===")

def make_lazy(func, *args, **kwargs):
    """创建延迟执行的函数"""
    def lazy():
        return func(*args, **kwargs)
    return lazy

def expensive_operation():
    print("执行昂贵操作")
    return "结果"

lazy_result = make_lazy(expensive_operation)
print("创建延迟函数（未执行）")
print(f"调用: {lazy_result()}")

print("\n=== 运行完成 ===")

