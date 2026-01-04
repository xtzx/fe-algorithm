"""限流测试"""

import time

import pytest

from http_kit.rate_limit import ConcurrencyLimiter, RateLimiter, handle_429


class TestRateLimiter:
    """令牌桶限流器测试"""

    def test_basic_acquire(self) -> None:
        limiter = RateLimiter(requests_per_second=10, burst=1)

        # 第一次应该立即成功
        assert limiter.try_acquire() is True

        # 立即再次获取应该失败
        assert limiter.try_acquire() is False

    def test_refill(self) -> None:
        limiter = RateLimiter(requests_per_second=10, burst=1)

        # 消耗令牌
        limiter.try_acquire()

        # 等待令牌恢复
        time.sleep(0.15)  # 0.1秒 = 1个令牌

        # 应该可以再次获取
        assert limiter.try_acquire() is True

    def test_burst(self) -> None:
        limiter = RateLimiter(requests_per_second=10, burst=5)

        # 应该可以连续获取 burst 次
        for i in range(5):
            assert limiter.try_acquire() is True

        # 超过 burst 后应该失败
        assert limiter.try_acquire() is False

    def test_blocking_acquire(self) -> None:
        limiter = RateLimiter(requests_per_second=100, burst=1)

        # 消耗令牌
        limiter.acquire()

        # 阻塞获取应该等待
        start = time.time()
        result = limiter.acquire(timeout=1.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed >= 0.005  # 至少等待一点时间


class TestConcurrencyLimiter:
    """并发限制器测试"""

    def test_context_manager(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=2)

        with limiter:
            # 在限制器内
            pass

        # 正常退出

    def test_acquire_release(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1)

        assert limiter.acquire(timeout=0.1) is True

        # 再次获取应该超时
        assert limiter.acquire(timeout=0.1) is False

        # 释放后可以再次获取
        limiter.release()
        assert limiter.acquire(timeout=0.1) is True


class TestHandle429:
    """429 处理测试"""

    def test_retry_after_seconds(self) -> None:
        # 模拟响应
        class MockResponse:
            headers = {"Retry-After": "30"}

        wait = handle_429(MockResponse())
        assert wait == 30.0

    def test_default_wait(self) -> None:
        class MockResponse:
            headers = {}

        wait = handle_429(MockResponse())
        assert wait == 60.0  # 默认值

    def test_max_wait(self) -> None:
        class MockResponse:
            headers = {"Retry-After": "999999"}

        wait = handle_429(MockResponse(), max_wait=300.0)
        assert wait == 300.0

