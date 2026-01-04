"""
命令行接口
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


async def crawl_command(args: argparse.Namespace) -> int:
    """执行爬取命令"""
    from scraper.dedup import HashSetDedup, UrlQueue
    from scraper.fetcher import RateLimitedFetcher
    from scraper.parser import HtmlParser, extract_links, filter_links
    from scraper.pipeline import JsonLineWriter, Pipeline
    from scraper.robots import RobotsChecker
    from scraper.state import FileState, StateManager

    print(f"Starting crawl: {args.url}")
    print(f"  Max pages: {args.max_pages}")
    print(f"  Delay: {args.delay}s")
    print(f"  Output: {args.output}")
    print()

    # 初始化组件
    state_manager = StateManager(FileState(args.state))
    queue = UrlQueue(HashSetDedup())
    parser = HtmlParser()

    # robots.txt 检查器
    robots_checker = None
    if args.respect_robots:
        robots_checker = RobotsChecker(user_agent=args.user_agent)

    # 数据管道
    pipeline = Pipeline([
        JsonLineWriter(args.output),
    ])

    # 添加起始 URL
    queue.add(args.url)
    state_manager.add_pending(args.url)

    pages_crawled = 0

    async with RateLimitedFetcher(
        requests_per_second=1.0 / args.delay,
        user_agent=args.user_agent,
    ) as fetcher:
        async with pipeline:
            while not queue.is_empty and pages_crawled < args.max_pages:
                url = queue.pop()
                if url is None:
                    break

                # 检查 robots.txt
                if robots_checker:
                    if not await robots_checker.is_allowed(url):
                        print(f"  Blocked by robots.txt: {url}")
                        continue

                # 获取页面
                print(f"  Fetching: {url}")
                result = await fetcher.fetch(url)

                if not result.success:
                    print(f"    Failed: {result.error}")
                    state_manager.mark_failed(url)
                    continue

                state_manager.mark_processed(url)
                pages_crawled += 1

                # 解析页面
                soup = parser.parse(result.html)
                title = parser.extract_text(soup, "title")

                # 保存数据
                item = {
                    "url": url,
                    "title": title,
                    "status_code": result.status_code,
                }
                await pipeline.process(item)
                state_manager.increment_items()

                # 提取链接
                links = extract_links(result.html, url)
                links = filter_links(
                    links,
                    args.url,
                    same_domain=not args.follow_external,
                )

                added = queue.add_many(links)
                state_manager.add_pending_many(links)

                print(f"    Title: {title[:50]}...")
                print(f"    Links found: {len(links)}, new: {added}")

    # 保存状态
    state_manager.save()

    print()
    print("Crawl completed!")
    print(f"  Pages crawled: {pages_crawled}")
    print(f"  Items saved: {pipeline.processed_count}")
    print(f"  Failed: {state_manager.failed_count}")

    return 0


async def check_robots_command(args: argparse.Namespace) -> int:
    """检查 robots.txt"""
    from scraper.robots import RobotsChecker

    print(f"Checking robots.txt for: {args.url}")
    print(f"  User-Agent: {args.user_agent}")

    checker = RobotsChecker(user_agent=args.user_agent)
    allowed = await checker.is_allowed(args.url)
    delay = await checker.get_crawl_delay(args.url)

    print()
    print(f"  Allowed: {'Yes' if allowed else 'No'}")
    if delay:
        print(f"  Crawl-Delay: {delay}s")

    return 0 if allowed else 1


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="scraper",
        description="生产级爬虫工具",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # crawl 命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取网页")
    crawl_parser.add_argument("url", help="起始 URL")
    crawl_parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=10,
        help="最大爬取页面数（默认: 10）",
    )
    crawl_parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="请求间隔秒数（默认: 1.0）",
    )
    crawl_parser.add_argument(
        "--output", "-o",
        default="items.jsonl",
        help="输出文件（默认: items.jsonl）",
    )
    crawl_parser.add_argument(
        "--state", "-s",
        default="crawl_state.json",
        help="状态文件（默认: crawl_state.json）",
    )
    crawl_parser.add_argument(
        "--user-agent", "-u",
        default="PythonScraper/1.0",
        help="User-Agent",
    )
    crawl_parser.add_argument(
        "--respect-robots",
        action="store_true",
        help="遵守 robots.txt",
    )
    crawl_parser.add_argument(
        "--follow-external",
        action="store_true",
        help="跟踪外部链接",
    )

    # check-robots 命令
    robots_parser = subparsers.add_parser("check-robots", help="检查 robots.txt")
    robots_parser.add_argument("url", help="要检查的 URL")
    robots_parser.add_argument(
        "--user-agent", "-u",
        default="*",
        help="User-Agent（默认: *）",
    )

    return parser


def main() -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "crawl":
        return asyncio.run(crawl_command(args))
    elif args.command == "check-robots":
        return asyncio.run(check_robots_command(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())

