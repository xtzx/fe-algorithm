"""
重试策略

支持：
- 指数退避
- 自定义重试条件
- 最大重试次数
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Type

import httpx


@dataclass
class RetryConfig:
    """
    重试配置

    Example:
        ```python
        config = RetryConfig(
            max_retries=3,
            backoff_factor=0.5,
            retry_on_status=[500, 502, 503, 504],
        )
        ```
    """

    # 最大重试次数
    max_retries: int = 3

    # 退避因子（用于指数退避）
    backoff_factor: float = 0.5

    # 最大等待时间（秒）
    max_backoff: float = 30.0

    # 是否添加抖动（避免惊群效应）
    jitter: bool = True

    # 需要重试的 HTTP 状态码
    retry_on_status: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # 需要重试的异常类型
    retry_on_exceptions: list[Type[Exception]] = field(
        default_factory=lambda: [
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
        ]
    )

    def get_wait_time(self, retry_count: int) -> float:
        """
        计算等待时间（指数退避 + 可选抖动）

        Args:
            retry_count: 当前重试次数（从 1 开始）

        Returns:
            等待时间（秒）

        公式: wait = min(backoff_factor * 2^retry_count, max_backoff)
        """
        wait = min(
            self.backoff_factor * (2**retry_count),
            self.max_backoff,
        )

        if self.jitter:
            # 添加 0-25% 的随机抖动
            wait = wait * (1 + random.random() * 0.25)

        return wait

    def should_retry_status(self, status_code: int) -> bool:
        """检查状态码是否应该重试"""
        return status_code in self.retry_on_status

    def should_retry_exception(self, exc: Exception) -> bool:
        """检查异常是否应该重试"""
        return isinstance(exc, tuple(self.retry_on_exceptions))


def retry_decorator(config: RetryConfig | None = None):
    """
    重试装饰器

    Example:
        ```python
        @retry_decorator(RetryConfig(max_retries=3))
        def fetch_data():
            return httpx.get("https://api.example.com/data")
        ```
    """
    import functools
    import time

    if config is None:
        config = RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            retry_count = 0

            while retry_count <= config.max_retries:
                try:
                    result = func(*args, **kwargs)

                    # 如果返回的是 Response，检查状态码
                    if isinstance(result, httpx.Response):
                        if config.should_retry_status(result.status_code):
                            if retry_count < config.max_retries:
                                retry_count += 1
                                wait_time = config.get_wait_time(retry_count)
                                time.sleep(wait_time)
                                continue

                    return result

                except Exception as e:
                    if config.should_retry_exception(e):
                        last_exception = e
                        if retry_count < config.max_retries:
                            retry_count += 1
                            wait_time = config.get_wait_time(retry_count)
                            time.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise

            if last_exception:
                raise last_exception

            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


async def async_retry_decorator(config: RetryConfig | None = None):
    """
    异步重试装饰器

    Example:
        ```python
        @async_retry_decorator(RetryConfig(max_retries=3))
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com/data")
        ```
    """
    import asyncio
    import functools

    if config is None:
        config = RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            retry_count = 0

            while retry_count <= config.max_retries:
                try:
                    result = await func(*args, **kwargs)

                    # 如果返回的是 Response，检查状态码
                    if isinstance(result, httpx.Response):
                        if config.should_retry_status(result.status_code):
                            if retry_count < config.max_retries:
                                retry_count += 1
                                wait_time = config.get_wait_time(retry_count)
                                await asyncio.sleep(wait_time)
                                continue

                    return result

                except Exception as e:
                    if config.should_retry_exception(e):
                        last_exception = e
                        if retry_count < config.max_retries:
                            retry_count += 1
                            wait_time = config.get_wait_time(retry_count)
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise

            if last_exception:
                raise last_exception

            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator

