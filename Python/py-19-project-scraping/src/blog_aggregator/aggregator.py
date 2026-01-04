"""
博客聚合器

协调多个源的采集
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from blog_aggregator.fetcher import Fetcher
from blog_aggregator.models import (
    AggregateStats,
    AppConfig,
    Article,
    ArticleList,
    CollectResult,
    SourceConfig,
)
from blog_aggregator.pipeline import Pipeline, create_default_pipeline
from blog_aggregator.sources import SOURCES, get_source
from blog_aggregator.storage import ArticleStorage, StateStorage


class BlogAggregator:
    """
    博客聚合器

    Example:
        ```python
        aggregator = BlogAggregator(config)

        # 采集所有源
        stats = await aggregator.collect_all()

        # 采集特定源
        stats = await aggregator.collect(sources=["dev_to"])

        # 增量采集
        stats = await aggregator.collect_all(incremental=True)
        ```
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        source_configs: dict[str, SourceConfig] | None = None,
        data_dir: str | Path = "data",
    ) -> None:
        """
        初始化聚合器

        Args:
            config: 应用配置
            source_configs: 源配置字典
            data_dir: 数据目录
        """
        self.config = config or AppConfig()
        self.source_configs = source_configs or {}
        self.data_dir = Path(data_dir)

        # 存储
        self.storage = ArticleStorage(self.data_dir / "articles.jsonl")
        self.state = StateStorage(self.data_dir / "state.json")

        # 管道
        self.pipeline = create_default_pipeline()

        # 获取器（在 collect 时创建）
        self._fetcher: Fetcher | None = None

    async def collect_all(
        self,
        incremental: bool = False,
        max_pages: int = 3,
    ) -> AggregateStats:
        """
        采集所有启用的源

        Args:
            incremental: 是否增量采集
            max_pages: 每个源的最大页数
        """
        enabled_sources = [
            name
            for name, cfg in self.source_configs.items()
            if cfg.enabled and name in SOURCES
        ]

        if not enabled_sources:
            # 使用默认源
            enabled_sources = list(SOURCES.keys())

        return await self.collect(
            sources=enabled_sources,
            incremental=incremental,
            max_pages=max_pages,
        )

    async def collect(
        self,
        sources: list[str],
        incremental: bool = False,
        max_pages: int = 3,
    ) -> AggregateStats:
        """
        采集指定源

        Args:
            sources: 源名称列表
            incremental: 是否增量采集
            max_pages: 每个源的最大页数
        """
        stats = AggregateStats()

        async with Fetcher(
            max_concurrent=self.config.max_concurrent,
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            user_agent=self.config.user_agent,
        ) as fetcher:
            self._fetcher = fetcher

            # 并发采集所有源
            tasks = [
                self._collect_source(name, incremental, max_pages)
                for name in sources
                if name in SOURCES
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, CollectResult):
                    stats.add_result(result)
                elif isinstance(result, Exception):
                    stats.add_result(
                        CollectResult(
                            source="unknown",
                            success=False,
                            error=str(result),
                        )
                    )

        return stats

    async def _collect_source(
        self,
        source_name: str,
        incremental: bool,
        max_pages: int,
    ) -> CollectResult:
        """采集单个源"""
        start_time = time.perf_counter()

        try:
            # 获取源类
            source_class = get_source(source_name)
            if source_class is None:
                return CollectResult(
                    source=source_name,
                    success=False,
                    error=f"Unknown source: {source_name}",
                )

            # 获取源配置
            source_config = self.source_configs.get(
                source_name,
                SourceConfig(base_url=""),
            )

            # 创建源实例
            assert self._fetcher is not None
            source = source_class(source_config, self._fetcher.get_client())

            # 获取文章
            all_articles = await source.fetch_all(max_pages=max_pages)

            # 增量过滤
            if incremental:
                last_collect = self.state.get_last_collect(source_name)
                if last_collect:
                    all_articles.articles = [
                        a
                        for a in all_articles.articles
                        if a.published_at and a.published_at > last_collect
                    ]

            # 通过管道处理
            processed = self.pipeline.process_many(all_articles.articles)

            # 保存新文章
            new_count = self.storage.save_many(processed)

            # 更新状态
            self.state.update_collect_time(source_name)
            self.state.add_stats(len(processed), new_count)

            elapsed = time.perf_counter() - start_time

            return CollectResult(
                source=source_name,
                success=True,
                articles_count=len(processed),
                new_count=new_count,
                elapsed=elapsed,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.state.add_error(source_name, str(e))

            return CollectResult(
                source=source_name,
                success=False,
                error=str(e),
                elapsed=elapsed,
            )

    def get_articles(
        self,
        source: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[Article]:
        """
        获取文章

        Args:
            source: 按源过滤
            tag: 按标签过滤
            limit: 限制数量
        """
        articles = self.storage.load_all()

        # 过滤
        if source:
            articles = [a for a in articles if a.source == source]

        if tag:
            tag_lower = tag.lower()
            articles = [
                a for a in articles if tag_lower in [t.lower() for t in a.tags]
            ]

        # 按发布时间排序
        articles.sort(
            key=lambda a: a.published_at or a.collected_at,
            reverse=True,
        )

        # 限制数量
        if limit:
            articles = articles[:limit]

        return articles

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        articles = self.storage.load_all()

        # 按源统计
        by_source: dict[str, int] = {}
        # 按标签统计
        by_tag: dict[str, int] = {}

        for article in articles:
            by_source[article.source] = by_source.get(article.source, 0) + 1
            for tag in article.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        # 排序标签
        top_tags = sorted(by_tag.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_articles": len(articles),
            "by_source": by_source,
            "top_tags": dict(top_tags),
            "state": self.state.state.to_dict(),
        }

