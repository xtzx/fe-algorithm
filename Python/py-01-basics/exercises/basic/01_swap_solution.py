#!/usr/bin/env python3
"""
练习 1 答案：变量交换
"""


def swap(a, b):
    """交换两个变量的值，返回 (b, a)"""
    return b, a


# 测试
if __name__ == "__main__":
    a, b = 10, 20
    print(f"交换前: a = {a}, b = {b}")

    a, b = swap(a, b)
    print(f"交换后: a = {a}, b = {b}")

    # 验证
    assert a == 20 and b == 10, "交换失败"
    print("✅ 测试通过")

    # 其他测试
    x, y = 1, 2
    x, y = swap(x, y)
    assert x == 2 and y == 1

    print("✅ 所有测试通过")

