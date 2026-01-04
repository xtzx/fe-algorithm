#!/usr/bin/env python3
"""
httpx 异步用法示例
"""

import asyncio
import time

import httpx


async def example_simple_async():
    """简单异步请求"""
    print("=" * 60)
    print("1. 简单异步请求")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("https://httpbin.org/get")
        print(f"Status: {response.status_code}")

    print()


async def example_concurrent_requests():
    """并发请求"""
    print("=" * 60)
    print("2. 并发请求")
    print("=" * 60)

    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
    ]

    async with httpx.AsyncClient() as client:
        # 串行请求
        start = time.time()
        for url in urls:
            await client.get(url)
        serial_time = time.time() - start
        print(f"串行请求: {serial_time:.2f}s")

        # 并发请求
        start = time.time()
        tasks = [client.get(url) for url in urls]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start
        print(f"并发请求: {concurrent_time:.2f}s")
        print(f"加速比: {serial_time / concurrent_time:.1f}x")

    print()


async def example_with_semaphore():
    """使用信号量限制并发"""
    print("=" * 60)
    print("3. 并发控制 (Semaphore)")
    print("=" * 60)

    semaphore = asyncio.Semaphore(2)  # 最多 2 个并发

    async def fetch_with_limit(client: httpx.AsyncClient, url: str):
        async with semaphore:
            print(f"Fetching: {url}")
            response = await client.get(url)
            print(f"Done: {url} - {response.status_code}")
            return response

    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
    ]

    async with httpx.AsyncClient() as client:
        start = time.time()
        tasks = [fetch_with_limit(client, url) for url in urls]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"总耗时: {elapsed:.2f}s (并发限制=2)")

    print()


async def example_streaming():
    """流式下载"""
    print("=" * 60)
    print("4. 流式下载")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "https://httpbin.org/stream-bytes/1024") as response:
            total_bytes = 0
            async for chunk in response.aiter_bytes(256):
                total_bytes += len(chunk)
                print(f"Received chunk: {len(chunk)} bytes")

            print(f"Total: {total_bytes} bytes")

    print()


async def example_error_handling():
    """异步错误处理"""
    print("=" * 60)
    print("5. 异步错误处理")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get("https://httpbin.org/status/500")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")

        try:
            response = await client.get("https://httpbin.org/delay/10")
        except httpx.ReadTimeout:
            print("Request timed out (expected)")

    print()


async def main():
    """运行所有异步示例"""
    print("\n" + "=" * 60)
    print("  httpx 异步用法示例")
    print("=" * 60 + "\n")

    await example_simple_async()
    await example_concurrent_requests()
    await example_with_semaphore()
    await example_streaming()
    await example_error_handling()

    print("=" * 60)
    print("  示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

