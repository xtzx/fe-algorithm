#!/usr/bin/env python3
"""
简单爬虫示例

演示基本的爬取流程:
1. 请求页面
2. 解析 HTML
3. 提取数据
4. 保存结果
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scraper import (
    RateLimitedFetcher,
    HtmlParser,
    Pipeline,
    JsonLineWriter,
    HashSetDedup,
    RobotsChecker,
    extract_links,
    filter_links,
)
from scraper.dedup import UrlQueue


async def simple_crawl(
    start_url: str,
    max_pages: int = 5,
    delay: float = 1.0,
    output_file: str = "output.jsonl",
):
    """
    简单爬虫

    Args:
        start_url: 起始 URL
        max_pages: 最大爬取页数
        delay: 请求间隔
        output_file: 输出文件
    """
    print(f"Starting crawl: {start_url}")
    print(f"  Max pages: {max_pages}")
    print(f"  Delay: {delay}s")
    print()

    # 初始化组件
    queue = UrlQueue(HashSetDedup())
    parser = HtmlParser()
    robots_checker = RobotsChecker()

    # 数据管道
    pipeline = Pipeline([
        JsonLineWriter(output_file, append=False),
    ])

    # 添加起始 URL
    queue.add(start_url)
    pages_crawled = 0
    items_saved = 0

    async with RateLimitedFetcher(
        requests_per_second=1.0 / delay,
        user_agent="SimpleCrawler/1.0 (Educational)",
    ) as fetcher:
        async with pipeline:
            while not queue.is_empty and pages_crawled < max_pages:
                url = queue.pop()
                if url is None:
                    break

                # 检查 robots.txt
                if not await robots_checker.is_allowed(url):
                    print(f"  [BLOCKED] {url} (robots.txt)")
                    continue

                # 获取页面
                print(f"  Fetching: {url}")
                result = await fetcher.fetch(url)

                if not result.success:
                    print(f"    [ERROR] {result.error}")
                    continue

                pages_crawled += 1

                # 解析页面
                soup = parser.parse(result.html)
                title = parser.extract_text(soup, "title")
                description = parser.extract_text(soup, "meta[name='description']") or ""

                # 保存数据
                item = {
                    "url": url,
                    "title": title,
                    "description": description[:200] if description else "",
                    "status_code": result.status_code,
                    "elapsed_ms": round(result.elapsed * 1000, 2),
                }

                await pipeline.process(item)
                items_saved += 1

                print(f"    [OK] {title[:50]}...")

                # 提取并添加链接
                links = extract_links(result.html, url)
                links = filter_links(links, start_url, same_domain=True)
                new_links = queue.add_many(links)

                print(f"    Links: {len(links)} found, {new_links} new")

    # 打印统计
    print()
    print("=" * 50)
    print("Crawl Summary")
    print("=" * 50)
    print(f"  Pages crawled: {pages_crawled}")
    print(f"  Items saved: {items_saved}")
    print(f"  URLs seen: {queue.seen_count}")
    print(f"  Output file: {output_file}")


async def demo_with_mock():
    """
    使用 Mock 数据演示（不需要网络）
    """
    print("=" * 50)
    print("Demo: 解析和去重功能")
    print("=" * 50)
    print()

    # 模拟 HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Demo Page</title>
        <meta name="description" content="This is a demo page">
    </head>
    <body>
        <h1>Welcome</h1>
        <a href="/page1">Page 1</a>
        <a href="/page2">Page 2</a>
        <a href="/page1">Page 1 Again</a>
    </body>
    </html>
    """
    base_url = "https://example.com"

    # 解析
    parser = HtmlParser()
    soup = parser.parse(html)

    title = parser.extract_text(soup, "title")
    print(f"Title: {title}")

    # 提取链接
    links = extract_links(html, base_url)
    print(f"Links found: {links}")

    # 去重
    queue = UrlQueue(HashSetDedup())
    new_count = queue.add_many(links)
    print(f"New links (after dedup): {new_count}")

    print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="简单爬虫示例")
    parser.add_argument(
        "url",
        nargs="?",
        default="",
        help="起始 URL（留空则运行 demo）",
    )
    parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=5,
        help="最大爬取页数",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="请求间隔（秒）",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
        help="输出文件",
    )

    args = parser.parse_args()

    if args.url:
        asyncio.run(simple_crawl(
            args.url,
            max_pages=args.max_pages,
            delay=args.delay,
            output_file=args.output,
        ))
    else:
        asyncio.run(demo_with_mock())


if __name__ == "__main__":
    main()

