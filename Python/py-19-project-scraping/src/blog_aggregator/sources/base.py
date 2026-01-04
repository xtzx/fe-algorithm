"""
博客源基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from blog_aggregator.models import Article, ArticleList, SourceConfig


class BaseSource(ABC):
    """
    博客源基类

    所有源解析器都应该继承此类
    """

    # 源名称
    name: str = "base"

    def __init__(self, config: SourceConfig, client: httpx.AsyncClient) -> None:
        """
        初始化源

        Args:
            config: 源配置
            client: HTTP 客户端（共享）
        """
        self.config = config
        self.client = client

    @abstractmethod
    async def fetch_articles(
        self,
        page: int = 1,
        per_page: int | None = None,
        tag: str | None = None,
    ) -> ArticleList:
        """
        获取文章列表

        Args:
            page: 页码
            per_page: 每页数量
            tag: 标签过滤

        Returns:
            ArticleList 对象
        """
        pass

    @abstractmethod
    def parse_article(self, data: dict[str, Any]) -> Article | None:
        """
        解析单篇文章

        Args:
            data: 原始数据

        Returns:
            Article 对象，解析失败返回 None
        """
        pass

    async def fetch_all(
        self,
        max_pages: int = 3,
        tag: str | None = None,
    ) -> ArticleList:
        """
        获取所有文章

        Args:
            max_pages: 最大页数
            tag: 标签过滤
        """
        all_articles = ArticleList(source=self.name)

        for page in range(1, max_pages + 1):
            articles = await self.fetch_articles(page=page, tag=tag)

            if not articles.articles:
                break

            for article in articles:
                all_articles.add(article)

        return all_articles

    def generate_id(self, *parts: str) -> str:
        """
        生成文章 ID

        格式: {source}_{parts...}
        """
        clean_parts = [str(p).replace("_", "-") for p in parts]
        return f"{self.name}_{'_'.join(clean_parts)}"

