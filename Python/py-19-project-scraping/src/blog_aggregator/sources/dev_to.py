"""
DEV.to 博客源解析器

DEV.to 是一个技术社区平台，提供 REST API
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from blog_aggregator.models import Article, ArticleList
from blog_aggregator.sources.base import BaseSource


class DevToSource(BaseSource):
    """
    DEV.to 源解析器

    API 文档: https://developers.forem.com/api
    """

    name = "dev_to"

    async def fetch_articles(
        self,
        page: int = 1,
        per_page: int | None = None,
        tag: str | None = None,
    ) -> ArticleList:
        """获取文章列表"""
        per_page = per_page or self.config.per_page
        api_url = self.config.api_url or "https://dev.to/api"

        params = {
            "page": page,
            "per_page": per_page,
        }

        if tag:
            params["tag"] = tag

        try:
            response = await self.client.get(
                f"{api_url}/articles",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            articles = ArticleList(source=self.name)
            for item in data:
                article = self.parse_article(item)
                if article:
                    articles.add(article)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching page {page}: {e}")
            return ArticleList(source=self.name)

    def parse_article(self, data: dict[str, Any]) -> Article | None:
        """解析文章"""
        try:
            # 解析发布时间
            published_at = None
            if data.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(
                        data["published_at"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # 提取作者信息
            user = data.get("user", {})

            return Article(
                id=self.generate_id(str(data.get("id", ""))),
                title=data.get("title", ""),
                url=data.get("url", ""),
                source=self.name,
                author=user.get("name", "") or user.get("username", ""),
                author_url=f"https://dev.to/{user.get('username', '')}",
                description=data.get("description", ""),
                cover_image=data.get("cover_image", "") or data.get("social_image", ""),
                tags=data.get("tag_list", []),
                reading_time=data.get("reading_time_minutes", 0),
                reactions=data.get("positive_reactions_count", 0),
                comments=data.get("comments_count", 0),
                published_at=published_at,
                raw_data=data,
            )

        except Exception as e:
            print(f"[{self.name}] Error parsing article: {e}")
            return None

    async def fetch_by_tag(self, tag: str, max_pages: int = 3) -> ArticleList:
        """按标签获取文章"""
        return await self.fetch_all(max_pages=max_pages, tag=tag)

    async def fetch_top_articles(self, top: int = 10) -> ArticleList:
        """获取热门文章"""
        api_url = self.config.api_url or "https://dev.to/api"

        try:
            response = await self.client.get(
                f"{api_url}/articles",
                params={"top": 7, "per_page": top},  # 最近 7 天热门
            )
            response.raise_for_status()
            data = response.json()

            articles = ArticleList(source=self.name)
            for item in data:
                article = self.parse_article(item)
                if article:
                    articles.add(article)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching top articles: {e}")
            return ArticleList(source=self.name)

