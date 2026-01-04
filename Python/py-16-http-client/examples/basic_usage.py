#!/usr/bin/env python3
"""
httpx 基础用法示例
"""

import httpx


def example_simple_request():
    """简单请求示例"""
    print("=" * 60)
    print("1. 简单 GET 请求")
    print("=" * 60)

    response = httpx.get("https://httpbin.org/get")
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)[:3]}...")
    print()


def example_with_client():
    """使用客户端实例"""
    print("=" * 60)
    print("2. 使用客户端实例")
    print("=" * 60)

    with httpx.Client(base_url="https://httpbin.org") as client:
        # GET
        response = client.get("/get", params={"name": "Alice"})
        print(f"GET: {response.status_code}")

        # POST JSON
        response = client.post("/post", json={"message": "Hello"})
        print(f"POST JSON: {response.status_code}")

        # POST Form
        response = client.post("/post", data={"username": "alice"})
        print(f"POST Form: {response.status_code}")

    print()


def example_headers():
    """自定义请求头"""
    print("=" * 60)
    print("3. 自定义请求头")
    print("=" * 60)

    with httpx.Client(
        base_url="https://httpbin.org",
        headers={
            "User-Agent": "MyApp/1.0",
            "Accept": "application/json",
        },
    ) as client:
        response = client.get("/headers")
        data = response.json()
        print(f"User-Agent: {data['headers'].get('User-Agent')}")
        print(f"Accept: {data['headers'].get('Accept')}")

    print()


def example_timeout():
    """超时配置"""
    print("=" * 60)
    print("4. 超时配置")
    print("=" * 60)

    timeout = httpx.Timeout(
        connect=5.0,
        read=10.0,
        write=10.0,
        pool=5.0,
    )

    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.get("https://httpbin.org/delay/1")
            print(f"Response: {response.status_code} (1s delay)")
        except httpx.TimeoutException:
            print("Request timed out")

    print()


def example_error_handling():
    """错误处理"""
    print("=" * 60)
    print("5. 错误处理")
    print("=" * 60)

    with httpx.Client() as client:
        # 404 错误
        response = client.get("https://httpbin.org/status/404")
        print(f"404 Response: {response.status_code}")

        # 使用 raise_for_status
        try:
            response = client.get("https://httpbin.org/status/500")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  httpx 基础用法示例")
    print("=" * 60 + "\n")

    example_simple_request()
    example_with_client()
    example_headers()
    example_timeout()
    example_error_handling()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

