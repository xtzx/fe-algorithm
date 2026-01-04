#!/usr/bin/env python3
"""pdb 调试演示

运行方式：
1. 直接运行：python pdb_demo.py
2. pdb 模式：python -m pdb pdb_demo.py

常用命令：
- n (next): 下一行
- s (step): 进入函数
- c (continue): 继续执行
- p expr: 打印表达式
- l (list): 显示代码
- w (where): 调用栈
- q (quit): 退出
"""


def calculate_sum(numbers: list[int]) -> int:
    """计算列表元素之和"""
    total = 0
    for n in numbers:
        total += n
    return total


def calculate_average(numbers: list[int]) -> float:
    """计算平均值

    注意：空列表会导致除零错误
    """
    total = calculate_sum(numbers)
    count = len(numbers)
    # 这里可能出错！
    return total / count


def find_max(numbers: list[int]) -> int:
    """查找最大值"""
    if not numbers:
        raise ValueError("列表不能为空")

    max_value = numbers[0]
    for n in numbers[1:]:
        if n > max_value:
            max_value = n
    return max_value


def buggy_function():
    """一个有 bug 的函数 - 演示调试"""
    data = [1, 2, 3, 4, 5]

    # 设置断点
    breakpoint()

    # 计算
    total = calculate_sum(data)
    avg = calculate_average(data)
    maximum = find_max(data)

    print(f"总和: {total}")
    print(f"平均: {avg}")
    print(f"最大: {maximum}")

    # 尝试空列表
    try:
        empty_avg = calculate_average([])
    except ZeroDivisionError as e:
        print(f"捕获错误: {e}")


def demo_conditional_breakpoint():
    """演示条件断点

    在 pdb 中使用：
    b 75, i > 5  # 当 i > 5 时在第 75 行断点
    """
    for i in range(10):
        # 可以设置条件断点
        result = i ** 2
        print(f"{i}^2 = {result}")


def demo_post_mortem():
    """演示事后调试

    运行：python -m pdb -c continue pdb_demo.py
    程序崩溃后自动进入调试器
    """
    x = 10
    y = 0
    result = x / y  # 这里会崩溃


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("pdb 调试演示")
    print("=" * 50)

    print("\n1. 基本调试演示")
    print("-" * 30)
    print("程序将在 breakpoint() 处暂停")
    print("使用 n/s/c/p 等命令调试")
    print()

    buggy_function()

    print("\n2. 条件断点演示")
    print("-" * 30)
    demo_conditional_breakpoint()

    # 取消注释以演示事后调试
    # print("\n3. 事后调试演示")
    # demo_post_mortem()

    print("\n演示完成!")

