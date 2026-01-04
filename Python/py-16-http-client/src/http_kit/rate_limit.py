"""
限流模块

支持：
- 请求速率限制
- 并发控制
- 429 处理
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """
    令牌桶限流器

    Example:
        ```python
        limiter = RateLimiter(requests_per_second=10)
        limiter.acquire()  # 阻塞直到获得令牌
        ```
    """

    # 每秒请求数
    requests_per_second: float = 10.0

    # 最大突发数量
    burst: int = 1

    # 内部状态
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._last_update = time.monotonic()

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # 计算应该补充的令牌数
        new_tokens = elapsed * self.requests_per_second
        self._tokens = min(self.burst, self._tokens + new_tokens)

    def acquire(self, timeout: float | None = None) -> bool:
        """
        获取令牌

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            是否成功获取令牌
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

                # 计算需要等待的时间
                wait_time = (1 - self._tokens) / self.requests_per_second

            # 检查是否超时
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            time.sleep(min(wait_time, 0.1))

    def try_acquire(self) -> bool:
        """
        尝试获取令牌（非阻塞）

        Returns:
            是否成功获取令牌
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            return False


class AsyncRateLimiter:
    """
    异步令牌桶限流器

    Example:
        ```python
        limiter = AsyncRateLimiter(requests_per_second=10)
        await limiter.acquire()  # 异步等待令牌
        ```
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst: int = 1,
    ) -> None:
        self.requests_per_second = requests_per_second
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        new_tokens = elapsed * self.requests_per_second
        self._tokens = min(self.burst, self._tokens + new_tokens)

    async def acquire(self, timeout: float | None = None) -> bool:
        """
        异步获取令牌

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否成功获取令牌
        """
        start_time = time.monotonic()

        while True:
            async with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

                wait_time = (1 - self._tokens) / self.requests_per_second

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            await asyncio.sleep(min(wait_time, 0.1))

    async def try_acquire(self) -> bool:
        """尝试获取令牌（非阻塞）"""
        async with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            return False


class ConcurrencyLimiter:
    """
    并发限制器

    Example:
        ```python
        limiter = ConcurrencyLimiter(max_concurrent=5)

        with limiter:
            # 最多 5 个并发
            do_something()
        ```
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)

    def __enter__(self) -> "ConcurrencyLimiter":
        self._semaphore.acquire()
        return self

    def __exit__(self, *args) -> None:
        self._semaphore.release()

    def acquire(self, timeout: float | None = None) -> bool:
        """获取并发槽位"""
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        """释放并发槽位"""
        self._semaphore.release()


class AsyncConcurrencyLimiter:
    """
    异步并发限制器

    Example:
        ```python
        limiter = AsyncConcurrencyLimiter(max_concurrent=5)

        async with limiter:
            await do_something()
        ```
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> "AsyncConcurrencyLimiter":
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        self._semaphore.release()


def handle_429(
    response,
    *,
    default_wait: float = 60.0,
    max_wait: float = 300.0,
) -> float:
    """
    处理 429 Too Many Requests

    从响应头中解析等待时间

    Args:
        response: HTTP 响应
        default_wait: 默认等待时间
        max_wait: 最大等待时间

    Returns:
        应该等待的时间（秒）
    """
    # 尝试从 Retry-After 头获取等待时间
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            # 如果是数字
            wait = float(retry_after)
            return min(wait, max_wait)
        except ValueError:
            pass

        try:
            # 如果是日期
            from email.utils import parsedate_to_datetime

            retry_date = parsedate_to_datetime(retry_after)
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            wait = (retry_date - now).total_seconds()
            return min(max(wait, 0), max_wait)
        except (ValueError, TypeError):
            pass

    # 尝试从 X-RateLimit-Reset 头获取
    rate_limit_reset = response.headers.get("X-RateLimit-Reset")
    if rate_limit_reset:
        try:
            reset_time = float(rate_limit_reset)
            wait = reset_time - time.time()
            return min(max(wait, 0), max_wait)
        except ValueError:
            pass

    return default_wait

