#!/usr/bin/env python3
"""内存与 GC 演示"""

import sys
import gc
import weakref
import tracemalloc


def demo_refcount():
    """引用计数演示"""
    print("=" * 50)
    print("1. 引用计数")
    print("=" * 50)

    a = [1, 2, 3]
    print(f"初始引用计数: {sys.getrefcount(a)}")  # 2 (a + getrefcount 参数)

    b = a  # 增加引用
    print(f"b = a 后: {sys.getrefcount(a)}")  # 3

    c = a  # 再增加
    print(f"c = a 后: {sys.getrefcount(a)}")  # 4

    del b  # 减少引用
    print(f"del b 后: {sys.getrefcount(a)}")  # 3

    del c
    print(f"del c 后: {sys.getrefcount(a)}")  # 2


def demo_id_is():
    """id() 和 is 演示"""
    print("\n" + "=" * 50)
    print("2. id() 和 is")
    print("=" * 50)

    # 不同对象
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(f"a == b: {a == b}")  # True (值相等)
    print(f"a is b: {a is b}")  # False (不同对象)
    print(f"id(a): {id(a)}, id(b): {id(b)}")

    # 同一对象
    c = a
    print(f"\nc = a 后:")
    print(f"a is c: {a is c}")  # True
    print(f"id(a): {id(a)}, id(c): {id(c)}")

    # 小整数缓存
    print(f"\n小整数缓存:")
    x = 100
    y = 100
    print(f"100 is 100: {x is y}")  # True

    x = 1000
    y = 1000
    print(f"1000 is 1000: {x is y}")  # 可能 False


def demo_gc():
    """垃圾回收演示"""
    print("\n" + "=" * 50)
    print("3. 垃圾回收")
    print("=" * 50)

    # 查看 GC 状态
    print(f"GC 启用: {gc.isenabled()}")
    print(f"GC 阈值: {gc.get_threshold()}")
    print(f"各代对象数: {gc.get_count()}")

    # 创建循环引用
    class Node:
        def __init__(self, name):
            self.name = name
            self.ref = None

    a = Node("A")
    b = Node("B")
    a.ref = b
    b.ref = a  # 循环引用

    # 删除变量
    del a, b

    # 手动 GC
    collected = gc.collect()
    print(f"\nGC 收集了 {collected} 个对象")


def demo_weakref():
    """弱引用演示"""
    print("\n" + "=" * 50)
    print("4. 弱引用")
    print("=" * 50)

    class MyClass:
        def __init__(self, name):
            self.name = name

        def __del__(self):
            print(f"  {self.name} 被删除")

    obj = MyClass("对象")

    # 创建弱引用
    weak = weakref.ref(obj)

    print(f"弱引用指向: {weak()}")
    print(f"弱引用增加引用计数: {sys.getrefcount(obj)}")  # 仍然是 2

    # 删除强引用
    print("删除强引用...")
    del obj

    print(f"删除后弱引用: {weak()}")  # None


def demo_tracemalloc():
    """内存追踪演示"""
    print("\n" + "=" * 50)
    print("5. tracemalloc 内存追踪")
    print("=" * 50)

    tracemalloc.start()

    # 分配内存
    data = [i ** 2 for i in range(100000)]

    # 获取快照
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("内存使用 Top 5:")
    for stat in top_stats[:5]:
        print(f"  {stat}")

    # 当前和峰值
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n当前内存: {current / 1024:.2f} KB")
    print(f"峰值内存: {peak / 1024:.2f} KB")

    tracemalloc.stop()

    # 清理
    del data


def demo_slots():
    """__slots__ 内存优化演示"""
    print("\n" + "=" * 50)
    print("6. __slots__ 内存优化")
    print("=" * 50)

    class WithoutSlots:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class WithSlots:
        __slots__ = ['x', 'y']

        def __init__(self, x, y):
            self.x = x
            self.y = y

    # 比较内存
    obj1 = WithoutSlots(1, 2)
    obj2 = WithSlots(1, 2)

    size1 = sys.getsizeof(obj1) + sys.getsizeof(obj1.__dict__)
    size2 = sys.getsizeof(obj2)

    print(f"无 __slots__: {size1} bytes")
    print(f"有 __slots__: {size2} bytes")
    print(f"节省: {(1 - size2/size1) * 100:.1f}%")


def demo_generator_memory():
    """生成器内存优势演示"""
    print("\n" + "=" * 50)
    print("7. 生成器 vs 列表 内存对比")
    print("=" * 50)

    # 列表：立即占用内存
    list_data = [i ** 2 for i in range(100000)]
    list_size = sys.getsizeof(list_data)

    # 生成器：按需生成
    gen_data = (i ** 2 for i in range(100000))
    gen_size = sys.getsizeof(gen_data)

    print(f"列表内存: {list_size:,} bytes")
    print(f"生成器内存: {gen_size:,} bytes")
    print(f"差距: {list_size / gen_size:.0f}x")

    del list_data


if __name__ == "__main__":
    demo_refcount()
    demo_id_is()
    demo_gc()
    demo_weakref()
    demo_tracemalloc()
    demo_slots()
    demo_generator_memory()

    print("\n✅ 内存与 GC 演示完成!")

