"""去重模块测试"""

import pytest
from pathlib import Path

from scraper import HashSetDedup, BloomFilter
from scraper.dedup import UrlQueue


class TestHashSetDedup:
    """HashSet 去重测试"""

    def test_add_new_url(self):
        dedup = HashSetDedup()
        assert dedup.add("https://example.com/page1") is True
        assert len(dedup) == 1

    def test_add_duplicate_url(self):
        dedup = HashSetDedup()
        dedup.add("https://example.com/page1")
        assert dedup.add("https://example.com/page1") is False
        assert len(dedup) == 1

    def test_contains(self):
        dedup = HashSetDedup()
        dedup.add("https://example.com/page1")

        assert dedup.contains("https://example.com/page1") is True
        assert dedup.contains("https://example.com/page2") is False

    def test_normalize_url(self):
        dedup = HashSetDedup()
        dedup.add("https://example.com/page#section")

        # 应该标准化（移除 fragment）
        assert dedup.contains("https://example.com/page") is True

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "dedup.json"

        # 保存
        dedup = HashSetDedup()
        dedup.add("https://example.com/page1")
        dedup.add("https://example.com/page2")
        dedup.save(path)

        # 加载
        loaded = HashSetDedup.load(path)
        assert len(loaded) == 2
        assert loaded.contains("https://example.com/page1")


class TestBloomFilter:
    """布隆过滤器测试"""

    def test_add_and_contains(self):
        bloom = BloomFilter(expected_items=1000)
        bloom.add("https://example.com/page1")

        assert bloom.contains("https://example.com/page1") is True

    def test_false_negative(self):
        bloom = BloomFilter(expected_items=1000)

        # 未添加的 URL 应该返回 False
        assert bloom.contains("https://example.com/nonexistent") is False

    def test_count(self):
        bloom = BloomFilter(expected_items=1000)
        bloom.add("https://example.com/page1")
        bloom.add("https://example.com/page2")

        assert len(bloom) == 2


class TestUrlQueue:
    """URL 队列测试"""

    def test_add_and_pop(self):
        queue = UrlQueue()
        queue.add("https://example.com/page1")
        queue.add("https://example.com/page2")

        assert len(queue) == 2
        assert queue.pop() == "https://example.com/page1"
        assert queue.pop() == "https://example.com/page2"
        assert queue.pop() is None

    def test_dedup(self):
        queue = UrlQueue()
        queue.add("https://example.com/page1")
        queue.add("https://example.com/page1")  # 重复

        assert len(queue) == 1

    def test_add_many(self):
        queue = UrlQueue()
        count = queue.add_many([
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page1",  # 重复
        ])

        assert count == 2
        assert len(queue) == 2

    def test_is_empty(self):
        queue = UrlQueue()
        assert queue.is_empty is True

        queue.add("https://example.com/page1")
        assert queue.is_empty is False

        queue.pop()
        assert queue.is_empty is True

    def test_seen_count(self):
        queue = UrlQueue()
        queue.add("https://example.com/page1")
        queue.pop()
        queue.add("https://example.com/page2")

        # seen_count 包括所有添加过的 URL
        assert queue.seen_count == 2
        # 但队列中只有 1 个
        assert len(queue) == 1

