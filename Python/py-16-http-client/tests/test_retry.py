"""重试策略测试"""

import pytest

from http_kit import HttpClient
from http_kit.retry import RetryConfig
from http_kit.testing import MockResponse, MockTransport


class TestRetryConfig:
    """重试配置测试"""

    def test_default_config(self) -> None:
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.backoff_factor == 0.5
        assert 429 in config.retry_on_status
        assert 500 in config.retry_on_status

    def test_wait_time_calculation(self) -> None:
        config = RetryConfig(backoff_factor=1.0, jitter=False)

        # 等待时间应该是指数增长
        assert config.get_wait_time(1) == 2.0  # 1.0 * 2^1
        assert config.get_wait_time(2) == 4.0  # 1.0 * 2^2
        assert config.get_wait_time(3) == 8.0  # 1.0 * 2^3

    def test_max_backoff(self) -> None:
        config = RetryConfig(backoff_factor=1.0, max_backoff=5.0, jitter=False)

        # 不应该超过 max_backoff
        assert config.get_wait_time(10) == 5.0

    def test_should_retry_status(self) -> None:
        config = RetryConfig(retry_on_status=[500, 503])

        assert config.should_retry_status(500) is True
        assert config.should_retry_status(503) is True
        assert config.should_retry_status(404) is False
        assert config.should_retry_status(200) is False


class TestRetryIntegration:
    """重试集成测试"""

    def test_retry_on_500(self) -> None:
        # 第一次 500，第二次成功
        transport = MockTransport([
            MockResponse(status_code=500),
            MockResponse(status_code=200, json_data={"ok": True}),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            retry_config=RetryConfig(max_retries=3, backoff_factor=0.01),
        )

        response = client.get("/unstable")

        assert response.status_code == 200
        assert len(transport.requests) == 2

    def test_retry_exhausted(self) -> None:
        # 所有请求都返回 500
        transport = MockTransport([
            MockResponse(status_code=500),
            MockResponse(status_code=500),
            MockResponse(status_code=500),
            MockResponse(status_code=500),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            retry_config=RetryConfig(max_retries=3, backoff_factor=0.01),
        )

        response = client.get("/always-fails")

        assert response.status_code == 500
        assert len(transport.requests) == 4  # 初始 + 3 次重试

    def test_no_retry_on_success(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200, json_data={"ok": True}),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            retry_config=RetryConfig(max_retries=3),
        )

        response = client.get("/stable")

        assert response.status_code == 200
        assert len(transport.requests) == 1

    def test_no_retry_on_client_error(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=404),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            retry_config=RetryConfig(max_retries=3),
        )

        response = client.get("/not-found")

        assert response.status_code == 404
        assert len(transport.requests) == 1  # 不重试 4xx

