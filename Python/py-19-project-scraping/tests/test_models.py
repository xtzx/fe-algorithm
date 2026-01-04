"""数据模型测试"""

import pytest
from datetime import datetime

from blog_aggregator.models import Article, ArticleList, CollectResult


class TestArticle:
    """Article 模型测试"""

    def test_create_article(self, sample_article_data):
        article = Article(**sample_article_data)

        assert article.id == "dev_to_12345"
        assert article.title == "Introduction to Python Async"
        assert article.source == "dev_to"
        assert len(article.tags) == 3

    def test_clean_title(self):
        article = Article(
            id="test_1",
            title="  Hello   World  ",
            url="https://example.com",
            source="test",
        )

        assert article.title == "Hello World"

    def test_clean_tags(self):
        article = Article(
            id="test_1",
            title="Test",
            url="https://example.com",
            source="test",
            tags=["Python", "  ASYNC  ", "web", ""],
        )

        assert article.tags == ["python", "async", "web"]

    def test_to_dict(self, sample_article_data):
        article = Article(**sample_article_data)
        data = article.to_dict()

        assert data["id"] == "dev_to_12345"
        assert isinstance(data["published_at"], str)
        assert isinstance(data["collected_at"], str)

    def test_from_dict(self, sample_article_data):
        # 先转换为字典格式
        article1 = Article(**sample_article_data)
        data = article1.to_dict()

        # 再从字典创建
        article2 = Article.from_dict(data)

        assert article2.id == article1.id
        assert article2.title == article1.title


class TestArticleList:
    """ArticleList 测试"""

    def test_add_article(self, sample_article_data):
        articles = ArticleList(source="dev_to")
        article = Article(**sample_article_data)

        articles.add(article)

        assert len(articles) == 1
        assert articles.total_count == 1

    def test_iterate(self, sample_article_data):
        articles = ArticleList(source="dev_to")
        article1 = Article(**sample_article_data)
        article2 = Article(**{**sample_article_data, "id": "dev_to_12346"})

        articles.add(article1)
        articles.add(article2)

        ids = [a.id for a in articles]
        assert ids == ["dev_to_12345", "dev_to_12346"]


class TestCollectResult:
    """CollectResult 测试"""

    def test_success_result(self):
        result = CollectResult(
            source="dev_to",
            success=True,
            articles_count=10,
            new_count=5,
            elapsed=2.5,
        )

        assert result.success
        assert result.articles_count == 10
        assert result.new_count == 5
        assert result.error is None

    def test_error_result(self):
        result = CollectResult(
            source="dev_to",
            success=False,
            error="Connection failed",
        )

        assert not result.success
        assert result.error == "Connection failed"

