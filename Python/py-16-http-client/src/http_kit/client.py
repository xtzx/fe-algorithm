"""
HTTP 客户端

提供同步和异步 HTTP 客户端，支持：
- 重试
- 限流
- 中间件
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from http_kit.rate_limit import RateLimiter
    from http_kit.retry import RetryConfig
    from http_kit.tracing import Middleware


class HttpClient:
    """
    同步 HTTP 客户端

    Example:
        ```python
        client = HttpClient(base_url="https://api.example.com")
        response = client.get("/users")
        users = response.json()
        ```
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        retry_config: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        middlewares: list[Middleware] | None = None,
        transport: httpx.BaseTransport | None = None,
        proxies: dict[str, str] | str | None = None,
        verify: bool = True,
    ) -> None:
        """
        初始化 HTTP 客户端

        Args:
            base_url: 基础 URL
            timeout: 默认超时时间（秒）
            headers: 默认请求头
            retry_config: 重试配置
            rate_limiter: 限流器
            middlewares: 中间件列表
            transport: 自定义传输层（用于测试）
            proxies: 代理配置
            verify: 是否验证 SSL 证书
        """
        self.base_url = base_url
        self.retry_config = retry_config
        self.rate_limiter = rate_limiter
        self.middlewares = middlewares or []

        # 构建 httpx 客户端配置
        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": httpx.Timeout(timeout),
            "headers": headers or {},
            "verify": verify,
        }

        if transport:
            client_kwargs["transport"] = transport

        if proxies:
            if isinstance(proxies, str):
                client_kwargs["proxies"] = {"all://": proxies}
            else:
                client_kwargs["proxies"] = proxies

        self._client = httpx.Client(**client_kwargs)

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """关闭客户端"""
        self._client.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        发送 HTTP 请求

        Args:
            method: HTTP 方法
            url: 请求 URL
            params: 查询参数
            json: JSON body
            data: 表单数据
            headers: 请求头
            timeout: 超时时间
            **kwargs: 其他参数

        Returns:
            HTTP 响应
        """
        # 应用限流
        if self.rate_limiter:
            self.rate_limiter.acquire()

        # 应用中间件 - 请求前
        request_headers = dict(headers or {})
        for middleware in self.middlewares:
            request_headers = middleware.before_request(method, url, request_headers)

        start_time = time.perf_counter()

        # 执行请求（带重试）
        response = self._execute_with_retry(
            method=method,
            url=url,
            params=params,
            json=json,
            data=data,
            headers=request_headers,
            timeout=timeout,
            **kwargs,
        )

        elapsed = time.perf_counter() - start_time

        # 应用中间件 - 请求后
        for middleware in self.middlewares:
            middleware.after_request(method, url, response, elapsed)

        return response

    def _execute_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """执行请求（带重试逻辑）"""
        if not self.retry_config:
            return self._client.request(method, url, **kwargs)

        last_exception: Exception | None = None
        retry_count = 0

        while retry_count <= self.retry_config.max_retries:
            try:
                response = self._client.request(method, url, **kwargs)

                # 检查是否需要重试（基于状态码）
                if response.status_code in self.retry_config.retry_on_status:
                    if retry_count < self.retry_config.max_retries:
                        retry_count += 1
                        wait_time = self.retry_config.get_wait_time(retry_count)
                        time.sleep(wait_time)
                        continue

                return response

            except tuple(self.retry_config.retry_on_exceptions) as e:
                last_exception = e
                if retry_count < self.retry_config.max_retries:
                    retry_count += 1
                    wait_time = self.retry_config.get_wait_time(retry_count)
                    time.sleep(wait_time)
                else:
                    raise

        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected retry loop exit")

    # =========================================================================
    # 便捷方法
    # =========================================================================

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """GET 请求"""
        return self.request("GET", url, params=params, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """POST 请求"""
        return self.request("POST", url, json=json, data=data, headers=headers, **kwargs)

    def put(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """PUT 请求"""
        return self.request("PUT", url, json=json, data=data, headers=headers, **kwargs)

    def patch(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """PATCH 请求"""
        return self.request("PATCH", url, json=json, data=data, headers=headers, **kwargs)

    def delete(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """DELETE 请求"""
        return self.request("DELETE", url, params=params, headers=headers, **kwargs)


class AsyncHttpClient:
    """
    异步 HTTP 客户端

    Example:
        ```python
        async with AsyncHttpClient(base_url="https://api.example.com") as client:
            response = await client.get("/users")
            users = response.json()
        ```
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        retry_config: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        middlewares: list[Middleware] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        proxies: dict[str, str] | str | None = None,
        verify: bool = True,
    ) -> None:
        """
        初始化异步 HTTP 客户端

        Args:
            base_url: 基础 URL
            timeout: 默认超时时间（秒）
            headers: 默认请求头
            retry_config: 重试配置
            rate_limiter: 限流器
            middlewares: 中间件列表
            transport: 自定义传输层（用于测试）
            proxies: 代理配置
            verify: 是否验证 SSL 证书
        """
        self.base_url = base_url
        self.retry_config = retry_config
        self.rate_limiter = rate_limiter
        self.middlewares = middlewares or []

        # 构建 httpx 客户端配置
        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": httpx.Timeout(timeout),
            "headers": headers or {},
            "verify": verify,
        }

        if transport:
            client_kwargs["transport"] = transport

        if proxies:
            if isinstance(proxies, str):
                client_kwargs["proxies"] = {"all://": proxies}
            else:
                client_kwargs["proxies"] = proxies

        self._client = httpx.AsyncClient(**client_kwargs)

    async def __aenter__(self) -> "AsyncHttpClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """关闭客户端"""
        await self._client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        发送异步 HTTP 请求
        """
        import asyncio

        # 应用限流
        if self.rate_limiter:
            await asyncio.to_thread(self.rate_limiter.acquire)

        # 应用中间件 - 请求前
        request_headers = dict(headers or {})
        for middleware in self.middlewares:
            request_headers = middleware.before_request(method, url, request_headers)

        start_time = time.perf_counter()

        # 执行请求（带重试）
        response = await self._execute_with_retry(
            method=method,
            url=url,
            params=params,
            json=json,
            data=data,
            headers=request_headers,
            timeout=timeout,
            **kwargs,
        )

        elapsed = time.perf_counter() - start_time

        # 应用中间件 - 请求后
        for middleware in self.middlewares:
            middleware.after_request(method, url, response, elapsed)

        return response

    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """执行请求（带重试逻辑）"""
        import asyncio

        if not self.retry_config:
            return await self._client.request(method, url, **kwargs)

        last_exception: Exception | None = None
        retry_count = 0

        while retry_count <= self.retry_config.max_retries:
            try:
                response = await self._client.request(method, url, **kwargs)

                # 检查是否需要重试（基于状态码）
                if response.status_code in self.retry_config.retry_on_status:
                    if retry_count < self.retry_config.max_retries:
                        retry_count += 1
                        wait_time = self.retry_config.get_wait_time(retry_count)
                        await asyncio.sleep(wait_time)
                        continue

                return response

            except tuple(self.retry_config.retry_on_exceptions) as e:
                last_exception = e
                if retry_count < self.retry_config.max_retries:
                    retry_count += 1
                    wait_time = self.retry_config.get_wait_time(retry_count)
                    await asyncio.sleep(wait_time)
                else:
                    raise

        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected retry loop exit")

    # =========================================================================
    # 便捷方法
    # =========================================================================

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """GET 请求"""
        return await self.request("GET", url, params=params, headers=headers, **kwargs)

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """POST 请求"""
        return await self.request(
            "POST", url, json=json, data=data, headers=headers, **kwargs
        )

    async def put(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """PUT 请求"""
        return await self.request(
            "PUT", url, json=json, data=data, headers=headers, **kwargs
        )

    async def patch(
        self,
        url: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """PATCH 请求"""
        return await self.request(
            "PATCH", url, json=json, data=data, headers=headers, **kwargs
        )

    async def delete(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """DELETE 请求"""
        return await self.request(
            "DELETE", url, params=params, headers=headers, **kwargs
        )

