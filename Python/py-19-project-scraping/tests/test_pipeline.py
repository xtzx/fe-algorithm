"""数据管道测试"""

import pytest

from blog_aggregator.models import Article
from blog_aggregator.pipeline import (
    Pipeline,
    clean_html,
    validate_url,
    validate_title,
    normalize_tags,
    truncate_description,
    filter_by_tags,
    create_default_pipeline,
)


@pytest.fixture
def sample_article():
    return Article(
        id="test_1",
        title="Test Article",
        url="https://example.com/article",
        source="test",
        description="<p>This is a &amp; test</p>",
        tags=["Python", "Web", "python"],  # 有重复
    )


class TestPipelineSteps:
    """管道步骤测试"""

    def test_clean_html(self, sample_article):
        result = clean_html(sample_article)

        assert "<p>" not in result.description
        assert "&amp;" not in result.description
        assert "This is a & test" in result.description

    def test_validate_url_valid(self, sample_article):
        result = validate_url(sample_article)
        assert result is not None

    def test_validate_url_invalid(self):
        article = Article(
            id="test_1",
            title="Test",
            url="not-a-url",
            source="test",
        )
        result = validate_url(article)
        assert result is None

    def test_validate_url_empty(self):
        article = Article(
            id="test_1",
            title="Test",
            url="",
            source="test",
        )
        result = validate_url(article)
        assert result is None

    def test_validate_title_valid(self, sample_article):
        result = validate_title(sample_article)
        assert result is not None

    def test_validate_title_short(self):
        article = Article(
            id="test_1",
            title="Hi",
            url="https://example.com",
            source="test",
        )
        result = validate_title(article)
        assert result is None

    def test_normalize_tags(self, sample_article):
        result = normalize_tags(sample_article)

        # 应该去重、小写、排序
        assert result.tags == ["python", "web"]

    def test_truncate_description(self):
        article = Article(
            id="test_1",
            title="Test",
            url="https://example.com",
            source="test",
            description="A" * 500,
        )

        truncate = truncate_description(100)
        result = truncate(article)

        assert len(result.description) == 100
        assert result.description.endswith("...")

    def test_filter_by_tags_match(self):
        article = Article(
            id="test_1",
            title="Test",
            url="https://example.com",
            source="test",
            tags=["python", "web"],
        )

        filter_step = filter_by_tags(["python", "javascript"])
        result = filter_step(article)

        assert result is not None

    def test_filter_by_tags_no_match(self):
        article = Article(
            id="test_1",
            title="Test",
            url="https://example.com",
            source="test",
            tags=["rust", "go"],
        )

        filter_step = filter_by_tags(["python", "javascript"])
        result = filter_step(article)

        assert result is None


class TestPipeline:
    """Pipeline 测试"""

    def test_process(self, sample_article):
        pipeline = Pipeline()
        pipeline.add_step(validate_url)
        pipeline.add_step(clean_html)

        result = pipeline.process(sample_article)

        assert result is not None
        assert pipeline.processed_count == 1
        assert pipeline.dropped_count == 0

    def test_process_drop(self):
        article = Article(
            id="test_1",
            title="Hi",  # 太短
            url="https://example.com",
            source="test",
        )

        pipeline = Pipeline()
        pipeline.add_step(validate_title)

        result = pipeline.process(article)

        assert result is None
        assert pipeline.dropped_count == 1

    def test_process_many(self, sample_article):
        articles = [
            sample_article,
            Article(
                id="test_2",
                title="Hi",  # 会被丢弃
                url="https://example.com",
                source="test",
            ),
        ]

        pipeline = create_default_pipeline()
        results = pipeline.process_many(articles)

        assert len(results) == 1
        assert results[0].id == "test_1"

