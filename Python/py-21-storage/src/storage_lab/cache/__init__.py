"""
缓存模块

提供:
- Redis 客户端
- 缓存装饰器
- 分布式锁
- 限流
"""

from storage_lab.cache.client import CacheClient, get_cache_client
from storage_lab.cache.decorators import cached, cache_aside
from storage_lab.cache.lock import DistributedLock, distributed_lock
from storage_lab.cache.rate_limit import RateLimiter, SlidingWindowRateLimiter

__all__ = [
    "CacheClient",
    "get_cache_client",
    "cached",
    "cache_aside",
    "DistributedLock",
    "distributed_lock",
    "RateLimiter",
    "SlidingWindowRateLimiter",
]


