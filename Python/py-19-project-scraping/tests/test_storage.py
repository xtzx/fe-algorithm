"""存储测试"""

import pytest
from datetime import datetime
from pathlib import Path

from blog_aggregator.models import Article
from blog_aggregator.storage import ArticleStorage, StateStorage


@pytest.fixture
def sample_article():
    return Article(
        id="test_1",
        title="Test Article",
        url="https://example.com/article",
        source="test",
        tags=["python"],
    )


class TestArticleStorage:
    """ArticleStorage 测试"""

    def test_save_and_load(self, tmp_path, sample_article):
        storage = ArticleStorage(tmp_path / "articles.jsonl")

        # 保存
        result = storage.save(sample_article)
        assert result is True
        assert storage.count == 1

        # 加载
        articles = storage.load_all()
        assert len(articles) == 1
        assert articles[0].id == "test_1"

    def test_save_duplicate(self, tmp_path, sample_article):
        storage = ArticleStorage(tmp_path / "articles.jsonl")

        # 第一次保存
        result1 = storage.save(sample_article)
        assert result1 is True

        # 第二次保存（重复）
        result2 = storage.save(sample_article)
        assert result2 is False
        assert storage.count == 1

    def test_exists(self, tmp_path, sample_article):
        storage = ArticleStorage(tmp_path / "articles.jsonl")
        storage.save(sample_article)

        assert storage.exists("test_1") is True
        assert storage.exists("test_999") is False

    def test_save_many(self, tmp_path):
        storage = ArticleStorage(tmp_path / "articles.jsonl")

        articles = [
            Article(
                id=f"test_{i}",
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                source="test",
            )
            for i in range(5)
        ]

        count = storage.save_many(articles)
        assert count == 5
        assert storage.count == 5

    def test_iter_articles(self, tmp_path, sample_article):
        storage = ArticleStorage(tmp_path / "articles.jsonl")
        storage.save(sample_article)

        articles = list(storage.iter_articles())
        assert len(articles) == 1

    def test_clear(self, tmp_path, sample_article):
        storage = ArticleStorage(tmp_path / "articles.jsonl")
        storage.save(sample_article)

        storage.clear()

        assert storage.count == 0
        assert not (tmp_path / "articles.jsonl").exists()


class TestStateStorage:
    """StateStorage 测试"""

    def test_update_collect_time(self, tmp_path):
        state = StateStorage(tmp_path / "state.json")

        state.update_collect_time("dev_to")

        last = state.get_last_collect("dev_to")
        assert last is not None
        assert isinstance(last, datetime)

    def test_get_last_collect_none(self, tmp_path):
        state = StateStorage(tmp_path / "state.json")

        last = state.get_last_collect("unknown")
        assert last is None

    def test_add_stats(self, tmp_path):
        state = StateStorage(tmp_path / "state.json")

        state.add_stats(10, 5)
        state.add_stats(20, 8)

        assert state.state.total_collected == 30
        assert state.state.total_new == 13

    def test_add_error(self, tmp_path):
        state = StateStorage(tmp_path / "state.json")

        state.add_error("dev_to", "Connection failed")

        assert len(state.state.errors) == 1
        assert state.state.errors[0]["source"] == "dev_to"

    def test_persistence(self, tmp_path):
        # 创建并保存
        state1 = StateStorage(tmp_path / "state.json")
        state1.update_collect_time("dev_to")
        state1.add_stats(10, 5)

        # 重新加载
        state2 = StateStorage(tmp_path / "state.json")

        assert state2.get_last_collect("dev_to") is not None
        assert state2.state.total_collected == 10

    def test_clear(self, tmp_path):
        state = StateStorage(tmp_path / "state.json")
        state.update_collect_time("dev_to")

        state.clear()

        assert state.get_last_collect("dev_to") is None

