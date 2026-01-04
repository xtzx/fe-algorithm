#!/usr/bin/env python3
"""
http_kit 高级功能示例
"""

import sys
import time

# 添加 src 到路径
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from http_kit import HttpClient
from http_kit.rate_limit import RateLimiter
from http_kit.retry import RetryConfig
from http_kit.testing import MockResponse, MockTransport
from http_kit.tracing import MetricsMiddleware, TracingMiddleware, set_trace_id


def example_retry():
    """重试示例"""
    print("=" * 60)
    print("1. 重试策略")
    print("=" * 60)

    # 模拟：前两次失败，第三次成功
    transport = MockTransport([
        MockResponse(status_code=503),
        MockResponse(status_code=503),
        MockResponse(status_code=200, json_data={"message": "Success!"}),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        retry_config=RetryConfig(
            max_retries=3,
            backoff_factor=0.1,  # 快速重试（演示）
        ),
    )

    print("发送请求（模拟前两次失败）...")
    response = client.get("/unstable")
    print(f"最终状态: {response.status_code}")
    print(f"重试次数: {len(transport.requests) - 1}")
    print(f"响应: {response.json()}")

    print()


def example_rate_limit():
    """限流示例"""
    print("=" * 60)
    print("2. 限流")
    print("=" * 60)

    transport = MockTransport([
        MockResponse(status_code=200) for _ in range(5)
    ])

    limiter = RateLimiter(requests_per_second=2, burst=1)

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        rate_limiter=limiter,
    )

    print("发送 5 个请求 (限制 2 RPS)...")
    start = time.time()

    for i in range(5):
        client.get(f"/item/{i}")
        print(f"  请求 {i + 1} 完成 @ {time.time() - start:.2f}s")

    elapsed = time.time() - start
    print(f"总耗时: {elapsed:.2f}s (预期 ~2s)")

    print()


def example_tracing():
    """追踪示例"""
    print("=" * 60)
    print("3. 追踪 (Trace ID)")
    print("=" * 60)

    transport = MockTransport([
        MockResponse(status_code=200),
        MockResponse(status_code=200),
    ])

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        middlewares=[TracingMiddleware()],
    )

    # 设置 trace_id
    set_trace_id("demo-trace-001")

    print("发送带 trace_id 的请求...")
    client.get("/users")
    client.get("/orders")

    # 验证请求头
    for i, request in enumerate(transport.requests):
        trace_id = request.headers.get("X-Trace-Id")
        print(f"  请求 {i + 1}: X-Trace-Id = {trace_id}")

    print()


def example_metrics():
    """指标收集示例"""
    print("=" * 60)
    print("4. 指标收集")
    print("=" * 60)

    transport = MockTransport([
        MockResponse(status_code=200),
        MockResponse(status_code=200),
        MockResponse(status_code=500),
        MockResponse(status_code=200),
        MockResponse(status_code=404),
    ])

    metrics = MetricsMiddleware()

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        middlewares=[metrics],
    )

    print("发送多个请求...")
    for i in range(5):
        client.get(f"/item/{i}")

    summary = metrics.get_summary()
    print(f"\n指标摘要:")
    print(f"  总请求数: {summary['total_requests']}")
    print(f"  平均延迟: {summary['avg_latency'] * 1000:.2f}ms")
    print(f"  错误率: {summary['error_rate']:.1f}%")
    print(f"  状态码分布: {summary['status_codes']}")

    print()


def example_full_featured():
    """完整功能示例"""
    print("=" * 60)
    print("5. 完整功能组合")
    print("=" * 60)

    # 模拟场景：第一次 503，第二次成功
    transport = MockTransport([
        MockResponse(status_code=503),
        MockResponse(status_code=200, json_data={"user": "Alice"}),
    ])

    metrics = MetricsMiddleware()
    tracing = TracingMiddleware()

    client = HttpClient(
        base_url="https://api.example.com",
        transport=transport,
        retry_config=RetryConfig(max_retries=2, backoff_factor=0.1),
        rate_limiter=RateLimiter(requests_per_second=10),
        middlewares=[tracing, metrics],
    )

    set_trace_id("full-demo-001")

    print("发送请求（带重试、限流、追踪、指标）...")
    response = client.get("/users/1")

    print(f"响应: {response.json()}")
    print(f"指标: {metrics.get_summary()}")

    print()


def main():
    """运行所有高级功能示例"""
    print("\n" + "=" * 60)
    print("  http_kit 高级功能示例")
    print("=" * 60 + "\n")

    example_retry()
    example_rate_limit()
    example_tracing()
    example_metrics()
    example_full_featured()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

