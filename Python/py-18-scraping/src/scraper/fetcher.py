"""
请求获取模块

支持:
- 静态页面抓取
- 频率限制
- 重试机制
- User-Agent 设置
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class FetchResult:
    """请求结果"""

    url: str
    status_code: int
    html: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    elapsed: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


class Fetcher:
    """
    HTTP 请求获取器

    Example:
        ```python
        async with Fetcher() as fetcher:
            result = await fetcher.fetch("https://example.com")
            print(result.html)
        ```
    """

    DEFAULT_HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (compatible; PythonScraper/1.0)",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
    ) -> None:
        """
        初始化请求获取器

        Args:
            user_agent: User-Agent 字符串
            timeout: 请求超时时间
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            headers: 额外请求头
            proxy: 代理地址
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proxy = proxy

        self._headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self._headers["User-Agent"] = user_agent

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Fetcher":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def start(self) -> None:
        """启动客户端"""
        client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self.timeout),
            "headers": self._headers,
            "follow_redirects": True,
        }

        if self.proxy:
            client_kwargs["proxies"] = {"all://": self.proxy}

        self._client = httpx.AsyncClient(**client_kwargs)

    async def close(self) -> None:
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        """
        获取页面

        Args:
            url: 目标 URL
            method: HTTP 方法
            headers: 额外请求头

        Returns:
            FetchResult 对象
        """
        if self._client is None:
            raise RuntimeError("Fetcher not started. Use 'async with' or call start()")

        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.perf_counter()

                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                )

                elapsed = time.perf_counter() - start_time

                return FetchResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    html=response.text,
                    headers=dict(response.headers),
                    elapsed=elapsed,
                )

            except httpx.TimeoutException:
                last_error = "Timeout"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
            except httpx.HTTPError as e:
                last_error = f"HTTP error: {e}"

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        return FetchResult(
            url=url,
            status_code=0,
            error=last_error,
        )


class RateLimitedFetcher(Fetcher):
    """
    带速率限制的请求获取器

    Example:
        ```python
        async with RateLimitedFetcher(requests_per_second=2) as fetcher:
            for url in urls:
                result = await fetcher.fetch(url)
        ```
    """

    def __init__(
        self,
        requests_per_second: float = 1.0,
        jitter: float = 0.5,
        **kwargs,
    ) -> None:
        """
        初始化带速率限制的获取器

        Args:
            requests_per_second: 每秒请求数
            jitter: 随机延迟范围（0-1，基于请求间隔的比例）
            **kwargs: 传递给 Fetcher 的参数
        """
        super().__init__(**kwargs)
        self.requests_per_second = requests_per_second
        self.jitter = jitter
        self._interval = 1.0 / requests_per_second
        self._last_request_time: float = 0.0

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        """带速率限制的获取"""
        # 计算需要等待的时间
        now = time.time()
        elapsed_since_last = now - self._last_request_time
        wait_time = self._interval - elapsed_since_last

        if wait_time > 0:
            # 添加随机抖动
            if self.jitter > 0:
                jitter_amount = self._interval * self.jitter * random.random()
                wait_time += jitter_amount

            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

        return await super().fetch(url, method=method, headers=headers)


class ProxyRotator:
    """
    代理轮换器

    Example:
        ```python
        rotator = ProxyRotator([
            "http://proxy1:8080",
            "http://proxy2:8080",
        ])

        proxy = rotator.get_next()
        ```
    """

    def __init__(self, proxies: list[str]) -> None:
        self.proxies = proxies
        self._index = 0
        self._failed: set[str] = set()

    def get_next(self) -> str | None:
        """获取下一个可用代理"""
        available = [p for p in self.proxies if p not in self._failed]
        if not available:
            return None

        self._index = (self._index + 1) % len(available)
        return available[self._index]

    def mark_failed(self, proxy: str) -> None:
        """标记代理为失败"""
        self._failed.add(proxy)

    def reset_failed(self) -> None:
        """重置失败列表"""
        self._failed.clear()

    @property
    def available_count(self) -> int:
        """可用代理数量"""
        return len(self.proxies) - len(self._failed)


async def fetch_with_proxy_rotation(
    url: str,
    proxies: list[str],
    max_retries: int = 3,
    **kwargs,
) -> FetchResult:
    """
    使用代理轮换获取页面

    Args:
        url: 目标 URL
        proxies: 代理列表
        max_retries: 最大重试次数
        **kwargs: 传递给 Fetcher 的参数
    """
    rotator = ProxyRotator(proxies)

    for _ in range(max_retries):
        proxy = rotator.get_next()
        if proxy is None:
            break

        async with Fetcher(proxy=proxy, **kwargs) as fetcher:
            result = await fetcher.fetch(url)

            if result.success:
                return result

            rotator.mark_failed(proxy)

    # 最后尝试不使用代理
    async with Fetcher(**kwargs) as fetcher:
        return await fetcher.fetch(url)

