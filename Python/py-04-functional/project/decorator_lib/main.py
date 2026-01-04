#!/usr/bin/env python3
"""
装饰器库演示

展示 @timer、@retry、@cache、@validate 的使用
"""

import time
import random
from decorators import timer, retry, cache, validate, is_positive, is_non_empty_string


# =============================================================================
# 1. @timer 演示
# =============================================================================
print("=" * 60)
print("1. @timer 装饰器演示")
print("=" * 60)

@timer
def slow_function():
    """慢速函数"""
    time.sleep(0.1)
    return "完成"

result = slow_function()
print(f"结果: {result}\n")


# =============================================================================
# 2. @retry 演示
# =============================================================================
print("=" * 60)
print("2. @retry 装饰器演示")
print("=" * 60)

@retry(max_attempts=3, delay=0.5)
def unstable_function():
    """不稳定的函数（可能失败）"""
    if random.random() < 0.7:
        raise ValueError("随机失败")
    return "成功"

try:
    result = unstable_function()
    print(f"结果: {result}\n")
except ValueError as e:
    print(f"最终失败: {e}\n")


# =============================================================================
# 3. @cache 演示
# =============================================================================
print("=" * 60)
print("3. @cache 装饰器演示")
print("=" * 60)

@cache
def expensive_function(n):
    """昂贵的计算函数"""
    print(f"计算 {n}...")
    time.sleep(0.1)
    return n ** 2

print("第一次调用:")
result1 = expensive_function(5)
print(f"结果: {result1}\n")

print("第二次调用（相同参数）:")
result2 = expensive_function(5)
print(f"结果: {result2}\n")

print("第三次调用（不同参数）:")
result3 = expensive_function(10)
print(f"结果: {result3}\n")

print("缓存信息:")
print(expensive_function.cache_info())


# =============================================================================
# 4. @validate 演示
# =============================================================================
print("\n" + "=" * 60)
print("4. @validate 装饰器演示")
print("=" * 60)

@validate(
    name=is_non_empty_string,
    age=is_positive
)
def create_user(name, age):
    """创建用户"""
    return {"name": name, "age": age}

print("✅ 有效参数:")
user1 = create_user("Alice", 25)
print(f"用户: {user1}\n")

print("❌ 无效参数:")
try:
    user2 = create_user("", 25)  # 空字符串
except ValueError as e:
    print(f"错误: {e}\n")

try:
    user3 = create_user("Bob", -5)  # 负数
except ValueError as e:
    print(f"错误: {e}\n")


# =============================================================================
# 5. 组合使用
# =============================================================================
print("=" * 60)
print("5. 组合使用多个装饰器")
print("=" * 60)

@timer
@cache
@retry(max_attempts=2, delay=0.1)
def complex_function(n):
    """复杂的函数（组合多个装饰器）"""
    if random.random() < 0.3:
        raise ValueError("随机失败")
    return n ** 2

print("第一次调用:")
result = complex_function(5)
print(f"结果: {result}\n")

print("第二次调用（缓存）:")
result = complex_function(5)
print(f"结果: {result}\n")


print("\n" + "=" * 60)
print("✅ 演示完成")
print("=" * 60)

