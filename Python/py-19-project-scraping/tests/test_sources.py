"""博客源测试"""

import pytest
import respx
import httpx

from blog_aggregator.models import SourceConfig
from blog_aggregator.sources.dev_to import DevToSource
from blog_aggregator.sources.hackernews import HackerNewsSource


class TestDevToSource:
    """DEV.to 源测试"""

    @respx.mock
    async def test_fetch_articles(self, sample_dev_to_response):
        # Mock API
        respx.get("https://dev.to/api/articles").respond(
            json=sample_dev_to_response
        )

        config = SourceConfig(base_url="https://dev.to", api_url="https://dev.to/api")

        async with httpx.AsyncClient() as client:
            source = DevToSource(config, client)
            articles = await source.fetch_articles(page=1, per_page=10)

        assert len(articles) == 2
        assert articles.articles[0].title == "Python Async Tutorial"
        assert articles.articles[0].source == "dev_to"

    @respx.mock
    async def test_parse_article(self, sample_dev_to_response):
        respx.get("https://dev.to/api/articles").respond(
            json=sample_dev_to_response
        )

        config = SourceConfig(base_url="https://dev.to", api_url="https://dev.to/api")

        async with httpx.AsyncClient() as client:
            source = DevToSource(config, client)
            article = source.parse_article(sample_dev_to_response[0])

        assert article is not None
        assert article.title == "Python Async Tutorial"
        assert article.author == "John Doe"
        assert "python" in article.tags
        assert article.reactions == 42


class TestHackerNewsSource:
    """Hacker News 源测试"""

    @respx.mock
    async def test_fetch_articles(self, sample_hackernews_item):
        api_url = "https://hacker-news.firebaseio.com/v0"

        # Mock 热门故事列表
        respx.get(f"{api_url}/topstories.json").respond(
            json=[38765432, 38765433]
        )

        # Mock 故事详情
        respx.get(f"{api_url}/item/38765432.json").respond(
            json=sample_hackernews_item
        )
        respx.get(f"{api_url}/item/38765433.json").respond(
            json={
                "id": 38765433,
                "type": "story",
                "title": "Another Story",
                "url": "https://example.com/story",
                "by": "user456",
                "time": 1705312800,
                "score": 50,
            }
        )

        config = SourceConfig(base_url="https://news.ycombinator.com", api_url=api_url)

        async with httpx.AsyncClient() as client:
            source = HackerNewsSource(config, client)
            articles = await source.fetch_articles(page=1, per_page=2)

        assert len(articles) == 2

    def test_parse_article(self, sample_hackernews_item):
        config = SourceConfig(base_url="https://news.ycombinator.com")

        # 不需要真正的 client 来测试解析
        import httpx

        source = HackerNewsSource(config, httpx.AsyncClient())
        article = source.parse_article(sample_hackernews_item)

        assert article is not None
        assert article.title == "Show HN: A new Python tool"
        assert article.source == "hackernews"
        assert article.reactions == 150
        assert article.comments == 45

