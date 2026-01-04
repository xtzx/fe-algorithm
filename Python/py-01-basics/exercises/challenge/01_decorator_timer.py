#!/usr/bin/env python3
"""
挑战练习：计时装饰器

题目：实现一个装饰器，打印函数执行时间

示例：
    @timer
    def slow_function():
        time.sleep(1)
        return "done"

    slow_function()  # 输出: slow_function 执行耗时: 1.00xx 秒
"""

import time
from functools import wraps


def timer(func):
    """计时装饰器"""
    # TODO: 实现装饰器逻辑
    pass


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

