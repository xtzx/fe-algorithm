"""
Hacker News 源解析器

使用 Firebase API
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from blog_aggregator.models import Article, ArticleList
from blog_aggregator.sources.base import BaseSource


class HackerNewsSource(BaseSource):
    """
    Hacker News 源解析器

    API 文档: https://github.com/HackerNews/API
    """

    name = "hackernews"

    async def fetch_articles(
        self,
        page: int = 1,
        per_page: int | None = None,
        tag: str | None = None,
    ) -> ArticleList:
        """获取文章列表"""
        per_page = per_page or self.config.per_page
        api_url = self.config.api_url or "https://hacker-news.firebaseio.com/v0"

        # 计算偏移量
        offset = (page - 1) * per_page

        try:
            # 获取热门故事 ID 列表
            response = await self.client.get(f"{api_url}/topstories.json")
            response.raise_for_status()
            story_ids = response.json()

            # 截取当前页的 ID
            page_ids = story_ids[offset : offset + per_page]

            if not page_ids:
                return ArticleList(source=self.name)

            # 并发获取故事详情
            articles = ArticleList(source=self.name)
            tasks = [self._fetch_story(api_url, story_id) for story_id in page_ids]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Article):
                    articles.add(result)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching articles: {e}")
            return ArticleList(source=self.name)

    async def _fetch_story(self, api_url: str, story_id: int) -> Article | None:
        """获取单个故事"""
        try:
            response = await self.client.get(f"{api_url}/item/{story_id}.json")
            response.raise_for_status()
            data = response.json()

            if data and data.get("type") == "story" and data.get("url"):
                return self.parse_article(data)

            return None

        except Exception:
            return None

    def parse_article(self, data: dict[str, Any]) -> Article | None:
        """解析文章"""
        try:
            # 解析发布时间（Unix 时间戳）
            published_at = None
            if data.get("time"):
                published_at = datetime.fromtimestamp(data["time"])

            # 估算阅读时间（基于标题长度，非常粗略）
            reading_time = 5  # 默认 5 分钟

            return Article(
                id=self.generate_id(str(data.get("id", ""))),
                title=data.get("title", ""),
                url=data.get("url", ""),
                source=self.name,
                author=data.get("by", ""),
                author_url=f"https://news.ycombinator.com/user?id={data.get('by', '')}",
                description="",  # HN 没有描述
                cover_image="",
                tags=[],  # HN 没有标签
                reading_time=reading_time,
                reactions=data.get("score", 0),
                comments=data.get("descendants", 0),
                published_at=published_at,
                raw_data=data,
            )

        except Exception as e:
            print(f"[{self.name}] Error parsing article: {e}")
            return None

    async def fetch_best_stories(self, limit: int = 30) -> ArticleList:
        """获取最佳故事"""
        api_url = self.config.api_url or "https://hacker-news.firebaseio.com/v0"

        try:
            response = await self.client.get(f"{api_url}/beststories.json")
            response.raise_for_status()
            story_ids = response.json()[:limit]

            articles = ArticleList(source=self.name)
            tasks = [self._fetch_story(api_url, sid) for sid in story_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Article):
                    articles.add(result)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching best stories: {e}")
            return ArticleList(source=self.name)

    async def fetch_new_stories(self, limit: int = 30) -> ArticleList:
        """获取最新故事"""
        api_url = self.config.api_url or "https://hacker-news.firebaseio.com/v0"

        try:
            response = await self.client.get(f"{api_url}/newstories.json")
            response.raise_for_status()
            story_ids = response.json()[:limit]

            articles = ArticleList(source=self.name)
            tasks = [self._fetch_story(api_url, sid) for sid in story_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Article):
                    articles.add(result)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching new stories: {e}")
            return ArticleList(source=self.name)

