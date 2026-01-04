"""
asyncio 并发学习库

提供:
- 基础并发模式
- 超时和取消
- 同步原语
- 实战模式
"""

from async_lab.basics import run_async, sleep_demo
from async_lab.concurrency import gather_with_errors, run_concurrently
from async_lab.patterns import ConcurrentExecutor, ProducerConsumer
from async_lab.stats import LatencyStats
from async_lab.timeout_cancel import with_timeout

__version__ = "0.1.0"

__all__ = [
    "run_async",
    "sleep_demo",
    "run_concurrently",
    "gather_with_errors",
    "with_timeout",
    "ProducerConsumer",
    "ConcurrentExecutor",
    "LatencyStats",
]

