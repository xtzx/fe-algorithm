"""测试配置和共享 fixture"""

import pytest

from http_kit.testing import AsyncMockTransport, MockResponse, MockTransport


@pytest.fixture
def mock_transport() -> MockTransport:
    """空的 Mock 传输层"""
    return MockTransport()


@pytest.fixture
def success_transport() -> MockTransport:
    """返回成功响应的传输层"""
    return MockTransport([
        MockResponse(status_code=200, json_data={"ok": True}),
    ])


@pytest.fixture
def async_mock_transport() -> AsyncMockTransport:
    """异步 Mock 传输层"""
    return AsyncMockTransport()

