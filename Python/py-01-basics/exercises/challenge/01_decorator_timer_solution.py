#!/usr/bin/env python3
"""
挑战练习答案：计时装饰器
"""

import time
from functools import wraps


def timer(func):
    """计时装饰器"""

    @wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result

    return wrapper


# 测试
if __name__ == "__main__":

    @timer
    def slow_function():
        """模拟耗时操作"""
        time.sleep(0.5)
        return "done"

    result = slow_function()
    print(f"返回值: {result}")

    # 验证函数名和文档字符串保留
    print(f"函数名: {slow_function.__name__}")
    print(f"文档: {slow_function.__doc__}")

    # 带参数的函数
    @timer
    def add(a, b):
        return a + b

    print(f"\nadd(1, 2) = {add(1, 2)}")

