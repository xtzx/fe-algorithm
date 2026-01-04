"""
统计工具

包含:
- 延迟统计
- 百分位数计算
- 请求统计
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


@dataclass
class LatencyStats:
    """
    延迟统计

    Example:
        ```python
        stats = LatencyStats()

        for _ in range(100):
            with stats.timer():
                await do_work()

        print(stats.summary())
        ```
    """

    _latencies: list[float] = field(default_factory=list)

    def record(self, latency: float) -> None:
        """记录一个延迟值"""
        self._latencies.append(latency)

    def timer(self) -> "LatencyTimer":
        """返回计时器上下文管理器"""
        return LatencyTimer(self)

    @property
    def count(self) -> int:
        """总记录数"""
        return len(self._latencies)

    @property
    def total(self) -> float:
        """总延迟"""
        return sum(self._latencies)

    @property
    def min(self) -> float:
        """最小延迟"""
        return min(self._latencies) if self._latencies else 0.0

    @property
    def max(self) -> float:
        """最大延迟"""
        return max(self._latencies) if self._latencies else 0.0

    @property
    def avg(self) -> float:
        """平均延迟"""
        return self.total / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """
        计算百分位数

        Args:
            p: 百分位（0-100）

        Returns:
            百分位值
        """
        if not self._latencies:
            return 0.0

        sorted_latencies = sorted(self._latencies)
        index = int(len(sorted_latencies) * p / 100)
        index = min(index, len(sorted_latencies) - 1)
        return sorted_latencies[index]

    @property
    def p50(self) -> float:
        """50 百分位（中位数）"""
        return self.percentile(50)

    @property
    def p90(self) -> float:
        """90 百分位"""
        return self.percentile(90)

    @property
    def p95(self) -> float:
        """95 百分位"""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99 百分位"""
        return self.percentile(99)

    def summary(self) -> dict[str, float]:
        """返回统计摘要"""
        return {
            "count": self.count,
            "total_ms": self.total * 1000,
            "avg_ms": self.avg * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "p50_ms": self.p50 * 1000,
            "p90_ms": self.p90 * 1000,
            "p95_ms": self.p95 * 1000,
            "p99_ms": self.p99 * 1000,
        }

    def clear(self) -> None:
        """清空记录"""
        self._latencies.clear()


class LatencyTimer:
    """延迟计时器"""

    def __init__(self, stats: LatencyStats) -> None:
        self._stats = stats
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._start
        self._stats.record(elapsed)

    async def __aenter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._start
        self._stats.record(elapsed)


@dataclass
class RequestStats:
    """
    请求统计

    跟踪成功、失败、重试等
    """

    total: int = 0
    success: int = 0
    failed: int = 0
    retried: int = 0
    timeout: int = 0
    latency: LatencyStats = field(default_factory=LatencyStats)

    def record_success(self, latency: float) -> None:
        """记录成功请求"""
        self.total += 1
        self.success += 1
        self.latency.record(latency)

    def record_failure(self, latency: float, is_timeout: bool = False) -> None:
        """记录失败请求"""
        self.total += 1
        self.failed += 1
        if is_timeout:
            self.timeout += 1
        self.latency.record(latency)

    def record_retry(self) -> None:
        """记录重试"""
        self.retried += 1

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success / self.total * 100 if self.total > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """失败率"""
        return self.failed / self.total * 100 if self.total > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """返回统计摘要"""
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "retried": self.retried,
            "timeout": self.timeout,
            "success_rate": f"{self.success_rate:.1f}%",
            "failure_rate": f"{self.failure_rate:.1f}%",
            "latency": self.latency.summary(),
        }


async def measure_async(
    func: Callable[..., Awaitable[T]],
    *args,
    **kwargs,
) -> tuple[T, float]:
    """
    测量异步函数执行时间

    Returns:
        (结果, 耗时秒数)
    """
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class ThroughputCounter:
    """
    吞吐量计数器

    计算每秒处理量

    Example:
        ```python
        counter = ThroughputCounter(window_seconds=5)

        for _ in range(1000):
            await process()
            counter.increment()

        print(f"Throughput: {counter.rate:.1f} ops/sec")
        ```
    """

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def increment(self, count: int = 1) -> None:
        """增加计数"""
        now = time.time()
        async with self._lock:
            for _ in range(count):
                self._timestamps.append(now)
            self._cleanup(now)

    def _cleanup(self, now: float) -> None:
        """清理过期的时间戳"""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)

    @property
    def rate(self) -> float:
        """当前速率（每秒）"""
        now = time.time()
        self._cleanup(now)
        if not self._timestamps:
            return 0.0
        return len(self._timestamps) / self.window_seconds

    @property
    def count(self) -> int:
        """窗口内计数"""
        now = time.time()
        self._cleanup(now)
        return len(self._timestamps)


def format_stats(stats: dict[str, Any], indent: int = 0) -> str:
    """
    格式化统计信息

    Args:
        stats: 统计字典
        indent: 缩进级别

    Returns:
        格式化的字符串
    """
    lines = []
    prefix = "  " * indent

    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_stats(value, indent + 1))
        elif isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.2f}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


async def benchmark(
    func: Callable[[], Awaitable[Any]],
    num_iterations: int = 100,
    warmup: int = 10,
) -> LatencyStats:
    """
    基准测试

    Args:
        func: 要测试的异步函数
        num_iterations: 迭代次数
        warmup: 预热次数

    Returns:
        延迟统计
    """
    # 预热
    for _ in range(warmup):
        await func()

    # 测试
    stats = LatencyStats()
    for _ in range(num_iterations):
        async with stats.timer():
            await func()

    return stats

