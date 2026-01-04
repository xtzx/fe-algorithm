#!/usr/bin/env python3
"""性能分析演示

运行方式：
1. 直接运行：python profile_demo.py
2. cProfile：python -m cProfile -s cumtime profile_demo.py
3. 生成统计：python -m cProfile -o profile.stats profile_demo.py
"""

import cProfile
import pstats
import io
import time
import timeit
from functools import lru_cache
from contextlib import contextmanager


# ============================================================
# 计时工具
# ============================================================

@contextmanager
def timer(name: str = "代码块"):
    """计时上下文管理器"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} 秒")


def profile_function(func):
    """分析装饰器"""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(10)
        print(stream.getvalue())

        return result
    return wrapper


# ============================================================
# 测试函数
# ============================================================

def slow_fibonacci(n: int) -> int:
    """慢速斐波那契（无缓存）"""
    if n < 2:
        return n
    return slow_fibonacci(n - 1) + slow_fibonacci(n - 2)


@lru_cache(maxsize=None)
def fast_fibonacci(n: int) -> int:
    """快速斐波那契（有缓存）"""
    if n < 2:
        return n
    return fast_fibonacci(n - 1) + fast_fibonacci(n - 2)


def slow_string_concat(n: int) -> str:
    """慢速字符串拼接"""
    result = ""
    for i in range(n):
        result += str(i)
    return result


def fast_string_concat(n: int) -> str:
    """快速字符串拼接"""
    return "".join(str(i) for i in range(n))


def slow_list_search(items: list, targets: list) -> list:
    """慢速列表查找"""
    found = []
    for target in targets:
        if target in items:  # O(n)
            found.append(target)
    return found


def fast_set_search(items: list, targets: list) -> list:
    """快速集合查找"""
    item_set = set(items)  # O(1) 查找
    found = []
    for target in targets:
        if target in item_set:
            found.append(target)
    return found


# ============================================================
# cProfile 演示
# ============================================================

def demo_cprofile():
    """cProfile 演示"""
    print("\n1. cProfile 演示")
    print("-" * 30)

    # 使用 cProfile.run()
    print("分析 slow_fibonacci(25):")
    cProfile.run("slow_fibonacci(25)", sort="cumtime")


def demo_pstats():
    """pstats 演示"""
    print("\n2. pstats 详细分析")
    print("-" * 30)

    # 收集统计
    profiler = cProfile.Profile()
    profiler.enable()

    result = sum(x ** 2 for x in range(100000))

    profiler.disable()

    # 分析
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    print("\n按累计时间排序 (前 5):")
    stats.print_stats(5)

    stats.sort_stats("tottime")
    print("\n按自身时间排序 (前 5):")
    stats.print_stats(5)


# ============================================================
# timeit 演示
# ============================================================

def demo_timeit():
    """timeit 微基准测试"""
    print("\n3. timeit 微基准测试")
    print("-" * 30)

    # 比较字符串拼接
    n = 1000

    time1 = timeit.timeit(
        lambda: slow_string_concat(n),
        number=100
    )

    time2 = timeit.timeit(
        lambda: fast_string_concat(n),
        number=100
    )

    print(f"字符串拼接 (n={n}, 100次):")
    print(f"  += 拼接: {time1:.4f}s")
    print(f"  join:   {time2:.4f}s")
    print(f"  加速:   {time1/time2:.1f}x")


# ============================================================
# 性能对比
# ============================================================

def demo_fibonacci_comparison():
    """斐波那契性能对比"""
    print("\n4. 斐波那契性能对比")
    print("-" * 30)

    n = 30

    # 慢速版本
    with timer(f"slow_fibonacci({n})"):
        result1 = slow_fibonacci(n)

    # 快速版本
    fast_fibonacci.cache_clear()
    with timer(f"fast_fibonacci({n})"):
        result2 = fast_fibonacci(n)

    print(f"结果: {result1} == {result2}")


def demo_search_comparison():
    """查找性能对比"""
    print("\n5. 查找性能对比")
    print("-" * 30)

    items = list(range(10000))
    targets = list(range(0, 10000, 10))  # 1000 个目标

    # 列表查找
    time1 = timeit.timeit(
        lambda: slow_list_search(items, targets),
        number=10
    )

    # 集合查找
    time2 = timeit.timeit(
        lambda: fast_set_search(items, targets),
        number=10
    )

    print(f"查找 1000 个元素 (10次):")
    print(f"  list: {time1:.4f}s")
    print(f"  set:  {time2:.4f}s")
    print(f"  加速: {time1/time2:.1f}x")


# ============================================================
# 装饰器分析
# ============================================================

def demo_decorator_profile():
    """使用装饰器分析"""
    print("\n6. 装饰器分析")
    print("-" * 30)

    @profile_function
    def compute_heavy():
        result = 0
        for i in range(100000):
            result += i ** 2
        return result

    compute_heavy()


# ============================================================
# 热点分析
# ============================================================

def inner_function(x):
    """内部函数（被频繁调用）"""
    return x ** 2


def outer_function(n):
    """外部函数"""
    result = 0
    for i in range(n):
        result += inner_function(i)
    return result


def demo_hotspot():
    """热点函数分析"""
    print("\n7. 热点函数分析")
    print("-" * 30)

    profiler = cProfile.Profile()
    profiler.enable()

    outer_function(100000)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    print("热点函数 (按自身时间):")
    stats.print_stats(5)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("性能分析演示")
    print("=" * 50)

    demo_cprofile()
    demo_pstats()
    demo_timeit()
    demo_fibonacci_comparison()
    demo_search_comparison()
    demo_decorator_profile()
    demo_hotspot()

    print("\n" + "=" * 50)
    print("提示：使用以下命令进行更详细的分析：")
    print("  python -m cProfile -s cumtime profile_demo.py")
    print("=" * 50)

