#!/usr/bin/env python3
"""
实用装饰器库

实现常用装饰器：
- @timer: 计时装饰器
- @retry: 重试装饰器
- @cache: 缓存装饰器
- @validate: 参数验证装饰器
"""

import time
import functools
from typing import Callable, Any, Dict, Tuple


def timer(func: Callable) -> Callable:
    """
    计时装饰器

    记录函数执行时间并打印

    Example:
        @timer
        def slow_function():
            time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            print(f"⏱️  {func.__name__} took {elapsed:.4f}s")
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: Tuple = (Exception,)):
    """
    重试装饰器

    Args:
        max_attempts: 最大尝试次数
        delay: 重试延迟（秒）
        exceptions: 需要重试的异常类型

    Example:
        @retry(max_attempts=3, delay=1)
        def unstable_function():
            if random.random() < 0.7:
                raise ValueError("随机失败")
            return "成功"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    print(f"⚠️  {func.__name__} 尝试 {attempt + 1}/{max_attempts} 失败: {e}")
                    print(f"    {delay}秒后重试...")
                    time.sleep(delay)

            print(f"❌ {func.__name__} 最终失败")
            raise last_exception
        return wrapper
    return decorator


def cache(func: Callable) -> Callable:
    """
    简单缓存装饰器

    缓存函数结果，相同参数直接返回缓存值

    Example:
        @cache
        def expensive_function(n):
            print(f"计算 {n}...")
            return n ** 2
    """
    cache_dict: Dict[Tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 创建缓存键
        key = (args, tuple(sorted(kwargs.items())))

        if key in cache_dict:
            print(f"💾 {func.__name__} 从缓存返回")
            return cache_dict[key]

        result = func(*args, **kwargs)
        cache_dict[key] = result
        print(f"💾 {func.__name__} 缓存结果")
        return result

    wrapper.cache_clear = lambda: cache_dict.clear()
    wrapper.cache_info = lambda: {
        "size": len(cache_dict),
        "keys": list(cache_dict.keys())
    }

    return wrapper


def validate(**validators: Callable):
    """
    参数验证装饰器

    Args:
        validators: 参数名 -> 验证函数的映射

    Example:
        @validate(
            name=lambda x: isinstance(x, str) and len(x) > 0,
            age=lambda x: isinstance(x, int) and 0 <= x <= 150
        )
        def create_user(name, age):
            return {"name": name, "age": age}
    """
    def decorator(func: Callable) -> Callable:
        import inspect
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 绑定参数
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"参数 {param_name} 验证失败: {value}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# 便捷验证器
def is_positive(x):
    """验证是否为正数"""
    return isinstance(x, (int, float)) and x > 0


def is_non_empty_string(x):
    """验证是否为非空字符串"""
    return isinstance(x, str) and len(x) > 0


def is_in_range(min_val, max_val):
    """创建范围验证器"""
    return lambda x: isinstance(x, (int, float)) and min_val <= x <= max_val

