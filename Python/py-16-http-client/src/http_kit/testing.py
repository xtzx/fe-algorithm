"""
测试工具

提供 Mock 传输层和测试辅助函数
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin

import httpx


@dataclass
class MockResponse:
    """Mock 响应配置"""

    # 匹配条件
    url: str | None = None
    method: str | None = None

    # 响应内容
    status_code: int = 200
    json_data: Any = None
    text: str | None = None
    content: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    # 动态响应
    callback: Callable[[httpx.Request], httpx.Response] | None = None

    def matches(self, request: httpx.Request) -> bool:
        """检查是否匹配请求"""
        if self.method and request.method != self.method.upper():
            return False

        if self.url:
            # 支持部分 URL 匹配
            request_path = str(request.url.path)
            if not request_path.endswith(self.url) and self.url not in str(request.url):
                return False

        return True

    def build_response(self, request: httpx.Request) -> httpx.Response:
        """构建响应"""
        if self.callback:
            return self.callback(request)

        # 确定响应内容
        if self.json_data is not None:
            content = json.dumps(self.json_data).encode()
            headers = {"content-type": "application/json", **self.headers}
        elif self.text is not None:
            content = self.text.encode()
            headers = {"content-type": "text/plain", **self.headers}
        elif self.content is not None:
            content = self.content
            headers = self.headers
        else:
            content = b""
            headers = self.headers

        return httpx.Response(
            status_code=self.status_code,
            headers=headers,
            content=content,
            request=request,
        )


class MockTransport(httpx.BaseTransport):
    """
    Mock 传输层

    用于测试 HTTP 客户端

    Example:
        ```python
        transport = MockTransport([
            MockResponse(url="/users", json_data=[{"id": 1, "name": "Alice"}]),
            MockResponse(url="/users/1", json_data={"id": 1, "name": "Alice"}),
        ])

        client = HttpClient(transport=transport)
        response = client.get("/users")
        ```
    """

    def __init__(
        self,
        responses: list[MockResponse | dict[str, Any]] | None = None,
        *,
        default_status: int = 404,
    ) -> None:
        """
        初始化 Mock 传输层

        Args:
            responses: 预设响应列表
            default_status: 未匹配时的默认状态码
        """
        self._responses: list[MockResponse] = []
        self.default_status = default_status
        self.requests: list[httpx.Request] = []  # 记录所有请求

        if responses:
            for r in responses:
                if isinstance(r, dict):
                    self._responses.append(MockResponse(**r))
                else:
                    self._responses.append(r)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """处理请求"""
        self.requests.append(request)

        # 查找匹配的响应
        for mock_response in self._responses:
            if mock_response.matches(request):
                return mock_response.build_response(request)

        # 返回默认响应
        return httpx.Response(
            status_code=self.default_status,
            content=b"Not Found",
            request=request,
        )

    def add_response(self, response: MockResponse | dict[str, Any]) -> None:
        """添加响应"""
        if isinstance(response, dict):
            response = MockResponse(**response)
        self._responses.append(response)

    def clear(self) -> None:
        """清空响应和请求记录"""
        self._responses.clear()
        self.requests.clear()

    def assert_called(self) -> None:
        """断言至少有一个请求"""
        assert len(self.requests) > 0, "No requests were made"

    def assert_called_once(self) -> None:
        """断言只有一个请求"""
        assert len(self.requests) == 1, f"Expected 1 request, got {len(self.requests)}"

    def assert_called_with(
        self,
        method: str | None = None,
        url: str | None = None,
    ) -> None:
        """断言最后一个请求匹配"""
        assert len(self.requests) > 0, "No requests were made"
        request = self.requests[-1]

        if method:
            assert request.method == method.upper(), f"Expected {method}, got {request.method}"

        if url:
            assert url in str(request.url), f"Expected URL containing {url}, got {request.url}"


class AsyncMockTransport(httpx.AsyncBaseTransport):
    """
    异步 Mock 传输层

    Example:
        ```python
        transport = AsyncMockTransport([
            MockResponse(url="/users", json_data=[{"id": 1}]),
        ])

        async with AsyncHttpClient(transport=transport) as client:
            response = await client.get("/users")
        ```
    """

    def __init__(
        self,
        responses: list[MockResponse | dict[str, Any]] | None = None,
        *,
        default_status: int = 404,
    ) -> None:
        self._sync_transport = MockTransport(responses, default_status=default_status)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """异步处理请求"""
        return self._sync_transport.handle_request(request)

    @property
    def requests(self) -> list[httpx.Request]:
        """获取请求记录"""
        return self._sync_transport.requests

    def add_response(self, response: MockResponse | dict[str, Any]) -> None:
        """添加响应"""
        self._sync_transport.add_response(response)

    def clear(self) -> None:
        """清空响应和请求记录"""
        self._sync_transport.clear()


def create_mock_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """
    创建 Mock 响应对象

    Example:
        ```python
        response = create_mock_response(
            status_code=200,
            json_data={"id": 1, "name": "Alice"},
        )
        ```
    """
    if json_data is not None:
        content = json.dumps(json_data).encode()
        headers = {"content-type": "application/json", **(headers or {})}
    elif text is not None:
        content = text.encode()
        headers = {"content-type": "text/plain", **(headers or {})}
    else:
        content = b""
        headers = headers or {}

    return httpx.Response(
        status_code=status_code,
        headers=headers,
        content=content,
    )

