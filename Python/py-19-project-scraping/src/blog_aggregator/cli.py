"""
命令行接口
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


def load_config(config_path: str | Path | None = None) -> tuple[dict, dict]:
    """加载配置文件"""
    from blog_aggregator.models import AppConfig, SourceConfig

    config_path = Path(config_path or "config.toml")

    if not config_path.exists():
        return {}, {}

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    # 解析通用配置
    general = data.get("general", {})
    app_config = AppConfig(**general)

    # 解析源配置
    source_configs = {}
    for name, cfg in data.get("sources", {}).items():
        source_configs[name] = SourceConfig(**cfg)

    return app_config.model_dump(), source_configs


async def collect_command(args: argparse.Namespace) -> int:
    """采集命令"""
    from blog_aggregator.aggregator import BlogAggregator
    from blog_aggregator.models import AppConfig, SourceConfig
    from blog_aggregator.reporter import Reporter

    print("=" * 60)
    print("  技术博客聚合器 - 采集")
    print("=" * 60)
    print()

    # 加载配置
    app_config_dict, source_configs_dict = load_config(args.config)
    app_config = AppConfig(**app_config_dict) if app_config_dict else AppConfig()
    source_configs = {
        name: SourceConfig(**cfg) if isinstance(cfg, dict) else cfg
        for name, cfg in source_configs_dict.items()
    }

    # 创建聚合器
    aggregator = BlogAggregator(
        config=app_config,
        source_configs=source_configs,
        data_dir=args.data_dir,
    )

    # 确定要采集的源
    if args.all:
        print("采集所有启用的源...")
    elif args.source:
        print(f"采集源: {args.source}")
    else:
        print("采集所有可用源...")

    print(f"增量模式: {'是' if args.incremental else '否'}")
    print(f"最大页数: {args.max_pages}")
    print()

    # 执行采集
    if args.source:
        stats = await aggregator.collect(
            sources=[args.source],
            incremental=args.incremental,
            max_pages=args.max_pages,
        )
    else:
        stats = await aggregator.collect_all(
            incremental=args.incremental,
            max_pages=args.max_pages,
        )

    # 输出结果
    print()
    print("采集结果:")
    print("-" * 40)

    for result in stats.results:
        status = "✓" if result.success else "✗"
        if result.success:
            print(
                f"  {status} {result.source}: "
                f"{result.articles_count} 篇 (新: {result.new_count}) "
                f"[{result.elapsed:.1f}s]"
            )
        else:
            print(f"  {status} {result.source}: {result.error}")

    print("-" * 40)
    print(f"总计: {stats.total_articles} 篇文章, {stats.new_articles} 篇新文章")
    print(f"耗时: {stats.total_elapsed:.1f} 秒")

    return 0


async def report_command(args: argparse.Namespace) -> int:
    """报告命令"""
    from blog_aggregator.aggregator import BlogAggregator
    from blog_aggregator.reporter import Reporter

    # 加载配置
    app_config_dict, source_configs_dict = load_config(args.config)

    # 创建聚合器
    aggregator = BlogAggregator(data_dir=args.data_dir)

    # 获取文章
    articles = aggregator.get_articles(
        source=args.source,
        tag=args.tag,
        limit=args.limit,
    )

    if not articles:
        print("没有找到文章")
        return 1

    # 生成报告
    reporter = Reporter(articles)

    if args.format == "terminal":
        print(reporter.to_terminal(max_articles=args.limit or 20))
    elif args.format == "markdown":
        if args.output:
            reporter.save(args.output, format="markdown")
            print(f"报告已保存到: {args.output}")
        else:
            print(reporter.to_markdown())
    elif args.format == "json":
        if args.output:
            reporter.save(args.output, format="json")
            print(f"报告已保存到: {args.output}")
        else:
            import json

            print(json.dumps(reporter.to_json(), indent=2, ensure_ascii=False))

    return 0


async def status_command(args: argparse.Namespace) -> int:
    """状态命令"""
    from blog_aggregator.aggregator import BlogAggregator

    aggregator = BlogAggregator(data_dir=args.data_dir)
    stats = aggregator.get_stats()

    print("=" * 60)
    print("  技术博客聚合器 - 状态")
    print("=" * 60)
    print()

    print(f"总文章数: {stats['total_articles']}")
    print()

    print("按来源分布:")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    print()

    print("热门标签 (Top 10):")
    for tag, count in list(stats["top_tags"].items())[:10]:
        print(f"  {tag}: {count}")
    print()

    state = stats.get("state", {})
    if state.get("last_collect"):
        print("最后采集时间:")
        for source, time in state["last_collect"].items():
            print(f"  {source}: {time}")

    return 0


async def clear_command(args: argparse.Namespace) -> int:
    """清空命令"""
    from blog_aggregator.aggregator import BlogAggregator

    aggregator = BlogAggregator(data_dir=args.data_dir)

    if args.yes:
        aggregator.storage.clear()
        aggregator.state.clear()
        print("数据已清空")
    else:
        confirm = input("确定要清空所有数据吗？(y/N): ")
        if confirm.lower() == "y":
            aggregator.storage.clear()
            aggregator.state.clear()
            print("数据已清空")
        else:
            print("已取消")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="blog-aggregator",
        description="技术博客聚合器",
    )

    parser.add_argument(
        "--config", "-c",
        default="config.toml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="数据目录",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # collect 命令
    collect_parser = subparsers.add_parser("collect", help="采集文章")
    collect_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="采集所有启用的源",
    )
    collect_parser.add_argument(
        "--source", "-s",
        help="指定源（如 dev_to, hashnode）",
    )
    collect_parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="增量采集（只抓新文章）",
    )
    collect_parser.add_argument(
        "--max-pages", "-p",
        type=int,
        default=3,
        help="每个源的最大页数",
    )

    # report 命令
    report_parser = subparsers.add_parser("report", help="生成报告")
    report_parser.add_argument(
        "--format", "-f",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="输出格式",
    )
    report_parser.add_argument(
        "--output", "-o",
        help="输出文件路径",
    )
    report_parser.add_argument(
        "--source", "-s",
        help="按源过滤",
    )
    report_parser.add_argument(
        "--tag", "-t",
        help="按标签过滤",
    )
    report_parser.add_argument(
        "--limit", "-l",
        type=int,
        help="限制数量",
    )

    # status 命令
    subparsers.add_parser("status", help="查看状态")

    # clear 命令
    clear_parser = subparsers.add_parser("clear", help="清空数据")
    clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认",
    )

    return parser


def main() -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "collect":
        return asyncio.run(collect_command(args))
    elif args.command == "report":
        return asyncio.run(report_command(args))
    elif args.command == "status":
        return asyncio.run(status_command(args))
    elif args.command == "clear":
        return asyncio.run(clear_command(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())

