#!/usr/bin/env python3
"""性能工具演示"""

import time
import timeit
import cProfile
import pstats
import io
from functools import wraps


def demo_perf_counter():
    """time.perf_counter 计时演示"""
    print("=" * 50)
    print("1. time.perf_counter 计时")
    print("=" * 50)

    start = time.perf_counter()

    # 执行代码
    total = sum(i ** 2 for i in range(100000))

    end = time.perf_counter()
    print(f"计算结果: {total}")
    print(f"耗时: {end - start:.6f} 秒")


def demo_timer_decorator():
    """计时装饰器演示"""
    print("\n" + "=" * 50)
    print("2. 计时装饰器")
    print("=" * 50)

    def timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"  {func.__name__} 耗时: {end - start:.6f}s")
            return result
        return wrapper

    @timer
    def slow_sum(n):
        return sum(i ** 2 for i in range(n))

    @timer
    def fast_sum(n):
        return n * (n - 1) * (2 * n - 1) // 6

    print("比较两种求和方式:")
    slow_sum(100000)
    fast_sum(100000)


def demo_timeit():
    """timeit 微基准测试演示"""
    print("\n" + "=" * 50)
    print("3. timeit 微基准测试")
    print("=" * 50)

    # 比较列表推导式和 map
    setup = "data = list(range(1000))"

    time1 = timeit.timeit(
        "[x * 2 for x in data]",
        setup=setup,
        number=10000
    )

    time2 = timeit.timeit(
        "list(map(lambda x: x * 2, data))",
        setup=setup,
        number=10000
    )

    print(f"列表推导式: {time1:.4f}s")
    print(f"map + lambda: {time2:.4f}s")
    print(f"列表推导式更快: {time1 < time2}")


def demo_string_concat():
    """字符串拼接性能演示"""
    print("\n" + "=" * 50)
    print("4. 字符串拼接性能对比")
    print("=" * 50)

    # 方法 1：+= 拼接
    def concat_plus():
        s = ""
        for i in range(1000):
            s += str(i)
        return s

    # 方法 2：join
    def concat_join():
        return "".join(str(i) for i in range(1000))

    # 方法 3：列表 + join
    def concat_list_join():
        parts = []
        for i in range(1000):
            parts.append(str(i))
        return "".join(parts)

    time1 = timeit.timeit(concat_plus, number=1000)
    time2 = timeit.timeit(concat_join, number=1000)
    time3 = timeit.timeit(concat_list_join, number=1000)

    print(f"+= 拼接: {time1:.4f}s")
    print(f"生成器 + join: {time2:.4f}s")
    print(f"列表 + join: {time3:.4f}s")


def demo_cprofile():
    """cProfile 性能分析演示"""
    print("\n" + "=" * 50)
    print("5. cProfile 性能分析")
    print("=" * 50)

    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    def main():
        results = []
        for i in range(20):
            results.append(fibonacci(i))
        return results

    # 创建分析器
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # 格式化输出
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

    print(s.getvalue())


def demo_local_vs_global():
    """局部变量 vs 全局变量性能"""
    print("\n" + "=" * 50)
    print("6. 局部变量 vs 全局变量")
    print("=" * 50)

    import math

    # 使用全局
    def use_global():
        total = 0
        for i in range(1000):
            total += math.sqrt(i)
        return total

    # 使用局部
    def use_local():
        sqrt = math.sqrt  # 缓存到局部
        total = 0
        for i in range(1000):
            total += sqrt(i)
        return total

    time1 = timeit.timeit(use_global, number=10000)
    time2 = timeit.timeit(use_local, number=10000)

    print(f"全局访问: {time1:.4f}s")
    print(f"局部变量: {time2:.4f}s")
    print(f"局部变量更快: {time1 > time2}")


def demo_list_vs_generator():
    """列表 vs 生成器性能"""
    print("\n" + "=" * 50)
    print("7. 列表 vs 生成器")
    print("=" * 50)

    # 列表：全部计算后再求和
    def with_list():
        return sum([i ** 2 for i in range(10000)])

    # 生成器：边计算边求和
    def with_generator():
        return sum(i ** 2 for i in range(10000))

    time1 = timeit.timeit(with_list, number=1000)
    time2 = timeit.timeit(with_generator, number=1000)

    print(f"列表推导式: {time1:.4f}s")
    print(f"生成器表达式: {time2:.4f}s")

    # 内存对比
    import sys
    list_data = [i ** 2 for i in range(10000)]
    gen_data = (i ** 2 for i in range(10000))

    print(f"\n内存占用:")
    print(f"  列表: {sys.getsizeof(list_data):,} bytes")
    print(f"  生成器: {sys.getsizeof(gen_data):,} bytes")


def demo_dict_methods():
    """字典访问方式性能"""
    print("\n" + "=" * 50)
    print("8. 字典访问方式对比")
    print("=" * 50)

    data = {str(i): i for i in range(1000)}

    # 直接访问（可能 KeyError）
    def direct_access():
        try:
            return data["999"]
        except KeyError:
            return None

    # get 方法
    def get_method():
        return data.get("999")

    # in 检查后访问
    def in_check():
        if "999" in data:
            return data["999"]
        return None

    time1 = timeit.timeit(direct_access, number=100000)
    time2 = timeit.timeit(get_method, number=100000)
    time3 = timeit.timeit(in_check, number=100000)

    print(f"try-except: {time1:.4f}s")
    print(f"dict.get(): {time2:.4f}s")
    print(f"in + 访问: {time3:.4f}s")


if __name__ == "__main__":
    demo_perf_counter()
    demo_timer_decorator()
    demo_timeit()
    demo_string_concat()
    demo_cprofile()
    demo_local_vs_global()
    demo_list_vs_generator()
    demo_dict_methods()

    print("\n✅ 性能工具演示完成!")

