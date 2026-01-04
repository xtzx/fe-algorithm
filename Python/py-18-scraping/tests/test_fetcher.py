"""请求模块测试"""

import pytest
import respx
import httpx

from scraper import Fetcher, RateLimitedFetcher
from scraper.fetcher import ProxyRotator


class TestFetcher:
    """Fetcher 测试"""

    @respx.mock
    async def test_fetch_success(self):
        respx.get("https://example.com/").respond(
            status_code=200,
            text="<html><body>Hello</body></html>",
        )

        async with Fetcher() as fetcher:
            result = await fetcher.fetch("https://example.com/")

        assert result.success
        assert result.status_code == 200
        assert "Hello" in result.html

    @respx.mock
    async def test_fetch_not_found(self):
        respx.get("https://example.com/404").respond(
            status_code=404,
            text="Not Found",
        )

        async with Fetcher() as fetcher:
            result = await fetcher.fetch("https://example.com/404")

        assert result.status_code == 404
        # 404 不算成功
        assert not result.success

    @respx.mock
    async def test_fetch_with_retry(self):
        # 第一次失败，第二次成功
        respx.get("https://example.com/").mock(
            side_effect=[
                httpx.ConnectError("Connection failed"),
                httpx.Response(200, text="Success"),
            ]
        )

        async with Fetcher(max_retries=3, retry_delay=0.01) as fetcher:
            result = await fetcher.fetch("https://example.com/")

        assert result.success
        assert "Success" in result.html

    @respx.mock
    async def test_fetch_all_retries_failed(self):
        respx.get("https://example.com/").mock(
            side_effect=httpx.ConnectError("Always fails")
        )

        async with Fetcher(max_retries=2, retry_delay=0.01) as fetcher:
            result = await fetcher.fetch("https://example.com/")

        assert not result.success
        assert result.error is not None

    @respx.mock
    async def test_custom_headers(self):
        route = respx.get("https://example.com/").respond(200)

        async with Fetcher(user_agent="TestBot/1.0") as fetcher:
            await fetcher.fetch("https://example.com/")

        # 检查请求头
        request = route.calls[0].request
        assert request.headers["User-Agent"] == "TestBot/1.0"


class TestRateLimitedFetcher:
    """带速率限制的 Fetcher 测试"""

    @respx.mock
    async def test_rate_limiting(self):
        import time

        respx.get("https://example.com/").respond(200)

        async with RateLimitedFetcher(
            requests_per_second=10,  # 100ms 间隔
            jitter=0,
        ) as fetcher:
            start = time.perf_counter()

            await fetcher.fetch("https://example.com/")
            await fetcher.fetch("https://example.com/")
            await fetcher.fetch("https://example.com/")

            elapsed = time.perf_counter() - start

        # 3 个请求，至少需要 200ms (2 个间隔)
        assert elapsed >= 0.18  # 留一点余量


class TestProxyRotator:
    """代理轮换测试"""

    def test_get_next(self):
        rotator = ProxyRotator([
            "http://proxy1:8080",
            "http://proxy2:8080",
        ])

        # 轮换获取代理
        proxy1 = rotator.get_next()
        proxy2 = rotator.get_next()

        assert proxy1 in ["http://proxy1:8080", "http://proxy2:8080"]
        assert proxy1 != proxy2 or len(rotator.proxies) == 1

    def test_mark_failed(self):
        rotator = ProxyRotator([
            "http://proxy1:8080",
            "http://proxy2:8080",
        ])

        rotator.mark_failed("http://proxy1:8080")
        assert rotator.available_count == 1

        # 获取的应该是未失败的代理
        proxy = rotator.get_next()
        assert proxy == "http://proxy2:8080"

    def test_all_failed(self):
        rotator = ProxyRotator(["http://proxy1:8080"])

        rotator.mark_failed("http://proxy1:8080")
        assert rotator.get_next() is None

    def test_reset_failed(self):
        rotator = ProxyRotator(["http://proxy1:8080"])

        rotator.mark_failed("http://proxy1:8080")
        rotator.reset_failed()

        assert rotator.available_count == 1

