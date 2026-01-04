#!/usr/bin/env python3
"""GIL 与并发演示"""

import threading
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def cpu_bound(n):
    """CPU 密集型任务"""
    total = 0
    for i in range(n):
        total += i * i
    return total


def io_bound(seconds):
    """I/O 密集型任务（模拟）"""
    time.sleep(seconds)
    return f"睡眠 {seconds} 秒完成"


def demo_gil_effect():
    """演示 GIL 对 CPU 密集型任务的影响"""
    print("=" * 50)
    print("1. GIL 对 CPU 密集型任务的影响")
    print("=" * 50)

    n = 5_000_000

    # 单线程
    start = time.perf_counter()
    cpu_bound(n)
    cpu_bound(n)
    single_time = time.perf_counter() - start
    print(f"单线程: {single_time:.4f}s")

    # 多线程
    start = time.perf_counter()
    t1 = threading.Thread(target=cpu_bound, args=(n,))
    t2 = threading.Thread(target=cpu_bound, args=(n,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    multi_thread_time = time.perf_counter() - start
    print(f"多线程: {multi_thread_time:.4f}s")

    print(f"\n结论: 多线程 {'更慢' if multi_thread_time > single_time else '更快'}")
    print("(GIL 导致 CPU 密集型任务无法真正并行)")


def demo_io_threading():
    """演示多线程处理 I/O 密集型任务"""
    print("\n" + "=" * 50)
    print("2. 多线程处理 I/O 密集型任务")
    print("=" * 50)

    tasks = [0.5] * 4  # 4 个 0.5 秒的任务

    # 单线程
    start = time.perf_counter()
    for t in tasks:
        io_bound(t)
    single_time = time.perf_counter() - start
    print(f"单线程: {single_time:.4f}s")

    # 多线程
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(io_bound, tasks))
    multi_thread_time = time.perf_counter() - start
    print(f"多线程: {multi_thread_time:.4f}s")

    print(f"\n结论: 多线程快 {single_time / multi_thread_time:.1f}x")
    print("(I/O 等待时 GIL 会释放，多线程有效)")


def demo_multiprocessing():
    """演示多进程处理 CPU 密集型任务"""
    print("\n" + "=" * 50)
    print("3. 多进程处理 CPU 密集型任务")
    print("=" * 50)

    n = 5_000_000
    tasks = [n] * 4

    # 单进程
    start = time.perf_counter()
    for t in tasks:
        cpu_bound(t)
    single_time = time.perf_counter() - start
    print(f"单进程: {single_time:.4f}s")

    # 多进程
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(cpu_bound, tasks))
    multi_process_time = time.perf_counter() - start
    print(f"多进程: {multi_process_time:.4f}s")

    print(f"\n结论: 多进程快 {single_time / multi_process_time:.1f}x")
    print("(每个进程有独立的 GIL，可以真正并行)")


def demo_thread_pool():
    """线程池演示"""
    print("\n" + "=" * 50)
    print("4. 线程池使用")
    print("=" * 50)

    def task(n):
        time.sleep(0.1)
        return n * 2

    with ThreadPoolExecutor(max_workers=4) as executor:
        # map 方式
        results = list(executor.map(task, range(8)))
        print(f"map 结果: {results}")

        # submit 方式
        future = executor.submit(task, 100)
        print(f"submit 结果: {future.result()}")


def demo_thread_safety():
    """线程安全演示"""
    print("\n" + "=" * 50)
    print("5. 线程安全问题")
    print("=" * 50)

    counter = 0

    def unsafe_increment():
        nonlocal counter
        for _ in range(100000):
            counter += 1

    # 不安全的多线程
    counter = 0
    threads = [threading.Thread(target=unsafe_increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"不安全计数器: {counter} (预期: 400000)")

    # 使用锁
    counter = 0
    lock = threading.Lock()

    def safe_increment():
        nonlocal counter
        for _ in range(100000):
            with lock:
                counter += 1

    threads = [threading.Thread(target=safe_increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"安全计数器: {counter} (预期: 400000)")


if __name__ == "__main__":
    demo_gil_effect()
    demo_io_threading()
    demo_multiprocessing()
    demo_thread_pool()
    demo_thread_safety()

    print("\n✅ GIL 与并发演示完成!")

