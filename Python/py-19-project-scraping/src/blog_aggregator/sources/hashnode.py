"""
Hashnode 博客源解析器

Hashnode 使用 GraphQL API
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from blog_aggregator.models import Article, ArticleList
from blog_aggregator.sources.base import BaseSource


class HashnodeSource(BaseSource):
    """
    Hashnode 源解析器

    使用 GraphQL API
    """

    name = "hashnode"

    # GraphQL 查询
    FEED_QUERY = """
    query Feed($first: Int!, $after: String) {
        feed(first: $first, after: $after, filter: { type: PERSONALIZED }) {
            edges {
                node {
                    id
                    title
                    brief
                    url
                    slug
                    coverImage {
                        url
                    }
                    author {
                        name
                        username
                        profilePicture
                    }
                    publishedAt
                    readTimeInMinutes
                    reactionCount
                    responseCount
                    tags {
                        name
                        slug
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """

    async def fetch_articles(
        self,
        page: int = 1,
        per_page: int | None = None,
        tag: str | None = None,
    ) -> ArticleList:
        """获取文章列表"""
        per_page = per_page or self.config.per_page
        api_url = self.config.api_url or "https://gql.hashnode.com"

        # 模拟分页（GraphQL 使用 cursor）
        # 简化处理：只获取第一页
        if page > 1:
            return ArticleList(source=self.name)

        try:
            response = await self.client.post(
                api_url,
                json={
                    "query": self.FEED_QUERY,
                    "variables": {
                        "first": per_page,
                        "after": None,
                    },
                },
            )

            # Hashnode 可能需要认证，这里做降级处理
            if response.status_code != 200:
                return await self._fetch_fallback()

            data = response.json()

            if "errors" in data:
                return await self._fetch_fallback()

            articles = ArticleList(source=self.name)
            edges = data.get("data", {}).get("feed", {}).get("edges", [])

            for edge in edges:
                node = edge.get("node", {})
                article = self.parse_article(node)
                if article:
                    articles.add(article)

            return articles

        except Exception as e:
            print(f"[{self.name}] Error fetching articles: {e}")
            return await self._fetch_fallback()

    async def _fetch_fallback(self) -> ArticleList:
        """
        降级方案：使用公开的博客 API

        当 GraphQL API 需要认证时使用
        """
        articles = ArticleList(source=self.name)

        # 尝试获取一些公开的技术博客
        try:
            # 这里可以替换为实际可用的 API
            # 示例：获取 Hashnode 的公开博客
            pass
        except Exception:
            pass

        return articles

    def parse_article(self, data: dict[str, Any]) -> Article | None:
        """解析文章"""
        try:
            # 解析发布时间
            published_at = None
            if data.get("publishedAt"):
                try:
                    published_at = datetime.fromisoformat(
                        data["publishedAt"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # 提取作者信息
            author = data.get("author", {})

            # 提取标签
            tags = [tag.get("name", "") for tag in data.get("tags", [])]

            # 提取封面图
            cover_image = ""
            if data.get("coverImage"):
                cover_image = data["coverImage"].get("url", "")

            return Article(
                id=self.generate_id(str(data.get("id", data.get("slug", "")))),
                title=data.get("title", ""),
                url=data.get("url", ""),
                source=self.name,
                author=author.get("name", "") or author.get("username", ""),
                author_url=f"https://hashnode.com/@{author.get('username', '')}",
                description=data.get("brief", ""),
                cover_image=cover_image,
                tags=tags,
                reading_time=data.get("readTimeInMinutes", 0),
                reactions=data.get("reactionCount", 0),
                comments=data.get("responseCount", 0),
                published_at=published_at,
                raw_data=data,
            )

        except Exception as e:
            print(f"[{self.name}] Error parsing article: {e}")
            return None

