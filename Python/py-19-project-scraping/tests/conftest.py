"""测试配置"""

import pytest
from datetime import datetime


@pytest.fixture
def sample_article_data():
    """示例文章数据"""
    return {
        "id": "dev_to_12345",
        "title": "Introduction to Python Async",
        "url": "https://dev.to/example/intro-python-async",
        "source": "dev_to",
        "author": "John Doe",
        "author_url": "https://dev.to/johndoe",
        "description": "Learn about async programming in Python...",
        "cover_image": "https://example.com/cover.jpg",
        "tags": ["python", "async", "tutorial"],
        "reading_time": 5,
        "reactions": 42,
        "comments": 10,
        "published_at": datetime(2024, 1, 15, 10, 30, 0),
    }


@pytest.fixture
def sample_dev_to_response():
    """DEV.to API 响应示例"""
    return [
        {
            "id": 12345,
            "title": "Python Async Tutorial",
            "description": "Learn async programming",
            "url": "https://dev.to/example/python-async",
            "cover_image": "https://example.com/cover.jpg",
            "tag_list": ["python", "async"],
            "user": {
                "name": "John Doe",
                "username": "johndoe",
            },
            "published_at": "2024-01-15T10:30:00Z",
            "reading_time_minutes": 5,
            "positive_reactions_count": 42,
            "comments_count": 10,
        },
        {
            "id": 12346,
            "title": "JavaScript Basics",
            "description": "Learn JS fundamentals",
            "url": "https://dev.to/example/js-basics",
            "tag_list": ["javascript", "tutorial"],
            "user": {
                "name": "Jane Smith",
                "username": "janesmith",
            },
            "published_at": "2024-01-14T09:00:00Z",
            "reading_time_minutes": 8,
            "positive_reactions_count": 100,
            "comments_count": 25,
        },
    ]


@pytest.fixture
def sample_hackernews_item():
    """Hacker News 条目示例"""
    return {
        "id": 38765432,
        "type": "story",
        "title": "Show HN: A new Python tool",
        "url": "https://github.com/example/tool",
        "by": "hacker123",
        "time": 1705312800,  # 2024-01-15
        "score": 150,
        "descendants": 45,
    }

