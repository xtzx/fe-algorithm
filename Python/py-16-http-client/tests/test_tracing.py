"""追踪测试"""

import pytest

from http_kit import HttpClient
from http_kit.testing import MockResponse, MockTransport
from http_kit.tracing import (
    MetricsMiddleware,
    TracingMiddleware,
    get_trace_id,
    set_trace_id,
)


class TestTraceId:
    """Trace ID 测试"""

    def test_get_trace_id_generates_new(self) -> None:
        # 重置
        set_trace_id("")

        trace_id = get_trace_id()
        assert trace_id is not None
        assert len(trace_id) == 8

    def test_set_and_get_trace_id(self) -> None:
        set_trace_id("test-123")
        assert get_trace_id() == "test-123"

    def test_trace_id_persistence(self) -> None:
        set_trace_id("persistent")

        # 多次调用应该返回相同值
        assert get_trace_id() == "persistent"
        assert get_trace_id() == "persistent"


class TestTracingMiddleware:
    """追踪中间件测试"""

    def test_adds_trace_id_header(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
        ])

        middleware = TracingMiddleware(trace_id_header="X-Trace-Id")

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[middleware],
        )

        set_trace_id("test-trace-123")
        client.get("/users")

        request = transport.requests[0]
        assert request.headers["X-Trace-Id"] == "test-trace-123"

    def test_custom_header_name(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
        ])

        middleware = TracingMiddleware(trace_id_header="X-Request-Id")

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[middleware],
        )

        set_trace_id("custom-123")
        client.get("/users")

        request = transport.requests[0]
        assert request.headers["X-Request-Id"] == "custom-123"


class TestMetricsMiddleware:
    """指标中间件测试"""

    def test_records_metrics(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
            MockResponse(status_code=200),
            MockResponse(status_code=500),
        ])

        metrics = MetricsMiddleware()

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[metrics],
        )

        client.get("/users")
        client.get("/products")
        client.get("/error")

        assert len(metrics.get_metrics()) == 3

    def test_get_summary(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
            MockResponse(status_code=200),
            MockResponse(status_code=500),
        ])

        metrics = MetricsMiddleware()

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[metrics],
        )

        client.get("/1")
        client.get("/2")
        client.get("/3")

        summary = metrics.get_summary()

        assert summary["total_requests"] == 3
        assert "avg_latency" in summary
        assert summary["status_codes"][200] == 2
        assert summary["status_codes"][500] == 1

    def test_error_rate(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
            MockResponse(status_code=500),
        ])

        metrics = MetricsMiddleware()

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[metrics],
        )

        client.get("/success")
        client.get("/error")

        summary = metrics.get_summary()
        assert summary["error_rate"] == 50.0

    def test_clear_metrics(self) -> None:
        transport = MockTransport([
            MockResponse(status_code=200),
        ])

        metrics = MetricsMiddleware()

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
            middlewares=[metrics],
        )

        client.get("/")
        assert len(metrics.get_metrics()) == 1

        metrics.clear()
        assert len(metrics.get_metrics()) == 0

