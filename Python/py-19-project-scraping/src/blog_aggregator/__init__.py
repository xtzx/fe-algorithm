"""
技术博客聚合器

综合项目：运用网络和并发知识完成数据采集
"""

from blog_aggregator.aggregator import BlogAggregator
from blog_aggregator.fetcher import Fetcher
from blog_aggregator.models import Article, ArticleList, CollectResult
from blog_aggregator.pipeline import Pipeline
from blog_aggregator.storage import ArticleStorage

__version__ = "0.1.0"

__all__ = [
    "BlogAggregator",
    "Fetcher",
    "Article",
    "ArticleList",
    "CollectResult",
    "Pipeline",
    "ArticleStorage",
]

