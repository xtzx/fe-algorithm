#!/usr/bin/env python3
"""内存分析演示

运行方式：
python memory_demo.py
"""

import gc
import sys
import tracemalloc
import weakref
from typing import Any


# ============================================================
# tracemalloc 基本使用
# ============================================================

def demo_tracemalloc_basic():
    """tracemalloc 基本使用"""
    print("\n1. tracemalloc 基本使用")
    print("-" * 30)

    # 开始追踪
    tracemalloc.start()

    # 分配内存
    data = [x ** 2 for x in range(100000)]

    # 获取快照
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("内存使用 Top 5:")
    for stat in top_stats[:5]:
        print(f"  {stat}")

    # 当前和峰值
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n当前内存: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存: {peak / 1024 / 1024:.2f} MB")

    tracemalloc.stop()

    # 清理
    del data


# ============================================================
# 比较内存快照
# ============================================================

def demo_compare_snapshots():
    """比较内存快照"""
    print("\n2. 比较内存快照")
    print("-" * 30)

    tracemalloc.start()

    # 第一个快照
    snapshot1 = tracemalloc.take_snapshot()

    # 分配内存
    leaked_data = []
    for i in range(100):
        leaked_data.append([0] * 1000)

    # 第二个快照
    snapshot2 = tracemalloc.take_snapshot()

    # 比较
    diff = snapshot2.compare_to(snapshot1, "lineno")

    print("内存增长 Top 5:")
    for stat in diff[:5]:
        print(f"  {stat}")

    tracemalloc.stop()


# ============================================================
# 对象大小
# ============================================================

def demo_object_sizes():
    """对象大小分析"""
    print("\n3. 对象大小")
    print("-" * 30)

    # 基本类型
    objects = {
        "int 0": 0,
        "int 1": 1,
        "int 大": 10 ** 100,
        "float": 3.14,
        "str 空": "",
        "str 短": "hello",
        "str 长": "x" * 1000,
        "list 空": [],
        "list 小": [1, 2, 3],
        "dict 空": {},
        "set 空": set(),
    }

    for name, obj in objects.items():
        size = sys.getsizeof(obj)
        print(f"  {name}: {size} bytes")


# ============================================================
# __slots__ 优化
# ============================================================

class PointWithoutSlots:
    """普通类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointWithSlots:
    """使用 __slots__ 的类"""
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y


def demo_slots():
    """__slots__ 内存优化"""
    print("\n4. __slots__ 内存优化")
    print("-" * 30)

    # 创建大量对象
    n = 100000

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    points_no_slots = [PointWithoutSlots(i, i) for i in range(n)]

    snapshot2 = tracemalloc.take_snapshot()

    del points_no_slots
    gc.collect()

    points_with_slots = [PointWithSlots(i, i) for i in range(n)]

    snapshot3 = tracemalloc.take_snapshot()

    # 比较
    diff1 = snapshot2.compare_to(snapshot1, "lineno")
    diff2 = snapshot3.compare_to(snapshot2, "lineno")

    size1 = sum(stat.size for stat in diff1)
    size2 = sum(stat.size for stat in diff2)

    print(f"  无 __slots__: {size1 / 1024 / 1024:.2f} MB")
    print(f"  有 __slots__: {size2 / 1024 / 1024:.2f} MB")
    print(f"  节省: {(1 - size2/size1) * 100:.1f}%")

    tracemalloc.stop()


# ============================================================
# 循环引用
# ============================================================

class Node:
    """可能产生循环引用的类"""
    def __init__(self, name):
        self.name = name
        self.ref = None

    def __del__(self):
        print(f"  Node '{self.name}' 被销毁")


def demo_circular_reference():
    """循环引用演示"""
    print("\n5. 循环引用")
    print("-" * 30)

    print("创建循环引用:")
    a = Node("A")
    b = Node("B")
    a.ref = b
    b.ref = a  # 循环引用！

    print(f"  A 引用计数: {sys.getrefcount(a) - 1}")
    print(f"  B 引用计数: {sys.getrefcount(b) - 1}")

    # 删除引用
    print("\n删除局部引用:")
    del a
    del b

    print("  对象仍然存在（循环引用）")

    # 强制 GC
    print("\n强制垃圾回收:")
    gc.collect()


# ============================================================
# 弱引用
# ============================================================

def demo_weak_reference():
    """弱引用演示"""
    print("\n6. 弱引用")
    print("-" * 30)

    class Data:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"Data({self.value})"

    # 创建对象
    obj = Data(42)
    print(f"  创建对象: {obj}")

    # 创建弱引用
    weak = weakref.ref(obj)
    print(f"  弱引用: {weak()}")

    # 删除强引用
    del obj
    print(f"  删除后: {weak()}")


# ============================================================
# 生成器 vs 列表
# ============================================================

def demo_generator_memory():
    """生成器内存对比"""
    print("\n7. 生成器 vs 列表")
    print("-" * 30)

    n = 1000000

    # 列表
    tracemalloc.start()
    list_data = [x ** 2 for x in range(n)]
    current, peak = tracemalloc.get_traced_memory()
    list_memory = peak
    tracemalloc.stop()
    del list_data

    # 生成器
    tracemalloc.start()
    gen_data = (x ** 2 for x in range(n))
    current, peak = tracemalloc.get_traced_memory()
    gen_memory = peak
    tracemalloc.stop()
    del gen_data

    print(f"  列表 (n={n}): {list_memory / 1024 / 1024:.2f} MB")
    print(f"  生成器 (n={n}): {gen_memory / 1024:.2f} KB")
    print(f"  节省: {(1 - gen_memory/list_memory) * 100:.1f}%")


# ============================================================
# 内存泄漏示例
# ============================================================

class LeakyCache:
    """有内存泄漏的缓存"""
    _cache: dict[str, Any] = {}  # 类变量，永不清理

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


class GoodCache:
    """改进的缓存"""
    def __init__(self, maxsize: int = 100):
        self._cache: dict[str, Any] = {}
        self._maxsize = maxsize

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._maxsize:
            # 简单的淘汰策略
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value


def demo_memory_leak():
    """内存泄漏示例"""
    print("\n8. 内存泄漏示例")
    print("-" * 30)

    print("  LeakyCache: 类变量永不清理")
    print("  GoodCache: 实例变量 + 大小限制")
    print("  使用 @lru_cache(maxsize=N) 更好")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("内存分析演示")
    print("=" * 50)

    demo_tracemalloc_basic()
    demo_compare_snapshots()
    demo_object_sizes()
    demo_slots()
    demo_circular_reference()
    demo_weak_reference()
    demo_generator_memory()
    demo_memory_leak()

    print("\n演示完成!")

