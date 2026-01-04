"""
并发获取器

支持:
- 异步并发
- 每站点并发限制
- 全局速率限制
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class FetchResult:
    """获取结果"""

    url: str
    status_code: int = 0
    content: str = ""
    json_data: dict | None = None
    headers: dict[str, str] = field(default_factory=dict)
    elapsed: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


class Fetcher:
    """
    并发获取器

    支持:
    - 全局并发限制
    - 每站点并发限制
    - 速率限制

    Example:
        ```python
        async with Fetcher(
            max_concurrent=10,
            per_host_limit=3,
            rate_limit=2.0,
        ) as fetcher:
            result = await fetcher.fetch("https://example.com")
        ```
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        per_host_limit: int = 3,
        rate_limit: float = 2.0,
        timeout: float = 30.0,
        user_agent: str = "BlogAggregator/1.0",
        max_retries: int = 3,
    ) -> None:
        """
        初始化获取器

        Args:
            max_concurrent: 全局最大并发数
            per_host_limit: 每站点最大并发数
            rate_limit: 每秒请求数
            timeout: 请求超时
            user_agent: User-Agent
            max_retries: 最大重试次数
        """
        self.max_concurrent = max_concurrent
        self.per_host_limit = per_host_limit
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries

        # 全局信号量
        self._global_semaphore: asyncio.Semaphore | None = None
        # 每站点信号量
        self._host_semaphores: dict[str, asyncio.Semaphore] = {}
        # 速率限制
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()

        # HTTP 客户端
        self._client: httpx.AsyncClient | None = None

        # 统计
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0

    async def __aenter__(self) -> "Fetcher":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def start(self) -> None:
        """启动"""
        self._global_semaphore = asyncio.Semaphore(self.max_concurrent)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    async def close(self) -> None:
        """关闭"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._client is None:
            raise RuntimeError("Fetcher not started")
        return self._client

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        **kwargs,
    ) -> FetchResult:
        """
        获取 URL

        Args:
            url: 目标 URL
            method: HTTP 方法
            **kwargs: 传递给 httpx 的参数
        """
        if self._client is None or self._global_semaphore is None:
            raise RuntimeError("Fetcher not started")

        # 提取主机名
        from urllib.parse import urlparse

        host = urlparse(url).netloc

        # 获取每站点信号量
        if host not in self._host_semaphores:
            self._host_semaphores[host] = asyncio.Semaphore(self.per_host_limit)

        # 应用限制
        async with self._global_semaphore:
            async with self._host_semaphores[host]:
                await self._apply_rate_limit()
                return await self._do_fetch(url, method, **kwargs)

    async def _apply_rate_limit(self) -> None:
        """应用速率限制"""
        if self.rate_limit <= 0:
            return

        async with self._rate_lock:
            now = time.time()
            interval = 1.0 / self.rate_limit
            elapsed = now - self._last_request_time
            wait_time = interval - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._last_request_time = time.time()

    async def _do_fetch(
        self,
        url: str,
        method: str,
        **kwargs,
    ) -> FetchResult:
        """执行请求"""
        assert self._client is not None

        last_error: str | None = None
        self._request_count += 1

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.perf_counter()

                response = await self._client.request(method, url, **kwargs)
                elapsed = time.perf_counter() - start_time

                # 尝试解析 JSON
                json_data = None
                if "application/json" in response.headers.get("content-type", ""):
                    try:
                        json_data = response.json()
                    except Exception:
                        pass

                self._success_count += 1

                return FetchResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    content=response.text,
                    json_data=json_data,
                    headers=dict(response.headers),
                    elapsed=elapsed,
                )

            except httpx.TimeoutException:
                last_error = "Timeout"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
            except httpx.HTTPError as e:
                last_error = f"HTTP error: {e}"
            except Exception as e:
                last_error = f"Error: {e}"

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        self._error_count += 1

        return FetchResult(
            url=url,
            error=last_error,
        )

    async def fetch_many(
        self,
        urls: list[str],
        method: str = "GET",
    ) -> list[FetchResult]:
        """批量获取"""
        tasks = [self.fetch(url, method) for url in urls]
        return await asyncio.gather(*tasks)

    @property
    def stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self._request_count,
            "successful": self._success_count,
            "failed": self._error_count,
            "success_rate": (
                f"{self._success_count / self._request_count * 100:.1f}%"
                if self._request_count > 0
                else "N/A"
            ),
        }

