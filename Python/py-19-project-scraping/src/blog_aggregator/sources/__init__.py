"""
博客源解析器

每个解析器负责:
1. 获取文章列表
2. 解析 HTML/API 响应
3. 转换为统一的 Article 模型
"""

from blog_aggregator.sources.base import BaseSource
from blog_aggregator.sources.dev_to import DevToSource
from blog_aggregator.sources.hackernews import HackerNewsSource
from blog_aggregator.sources.hashnode import HashnodeSource

# 注册所有源
SOURCES: dict[str, type[BaseSource]] = {
    "dev_to": DevToSource,
    "hashnode": HashnodeSource,
    "hackernews": HackerNewsSource,
}


def get_source(name: str) -> type[BaseSource] | None:
    """获取源类"""
    return SOURCES.get(name)


def get_all_sources() -> dict[str, type[BaseSource]]:
    """获取所有源"""
    return SOURCES.copy()


__all__ = [
    "BaseSource",
    "DevToSource",
    "HashnodeSource",
    "HackerNewsSource",
    "SOURCES",
    "get_source",
    "get_all_sources",
]

