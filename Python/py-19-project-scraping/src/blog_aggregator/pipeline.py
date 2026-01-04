"""
数据管道

支持:
- 数据清洗
- 数据验证
- 去重
"""

from __future__ import annotations

import re
from typing import Any, Callable

from blog_aggregator.models import Article


class Pipeline:
    """
    数据处理管道

    Example:
        ```python
        pipeline = Pipeline()
        pipeline.add_step(clean_html)
        pipeline.add_step(validate_url)

        processed = pipeline.process(article)
        ```
    """

    def __init__(self) -> None:
        self._steps: list[Callable[[Article], Article | None]] = []
        self._processed_count = 0
        self._dropped_count = 0

    def add_step(self, step: Callable[[Article], Article | None]) -> "Pipeline":
        """添加处理步骤"""
        self._steps.append(step)
        return self

    def process(self, article: Article) -> Article | None:
        """
        处理文章

        Returns:
            处理后的文章，如果被丢弃则返回 None
        """
        result = article

        for step in self._steps:
            result = step(result)
            if result is None:
                self._dropped_count += 1
                return None

        self._processed_count += 1
        return result

    def process_many(self, articles: list[Article]) -> list[Article]:
        """批量处理"""
        results = []
        for article in articles:
            processed = self.process(article)
            if processed:
                results.append(processed)
        return results

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count


# =============================================================================
# 内置处理步骤
# =============================================================================


def clean_html(article: Article) -> Article:
    """清理 HTML 标签"""
    # 清理描述中的 HTML
    if article.description:
        # 移除 HTML 标签
        clean_desc = re.sub(r"<[^>]+>", "", article.description)
        # 处理 HTML 实体
        clean_desc = (
            clean_desc.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
        )
        # 规范化空白
        clean_desc = " ".join(clean_desc.split())
        article = article.model_copy(update={"description": clean_desc})

    return article


def validate_url(article: Article) -> Article | None:
    """验证 URL"""
    if not article.url:
        return None

    if not article.url.startswith(("http://", "https://")):
        return None

    return article


def validate_title(article: Article) -> Article | None:
    """验证标题"""
    if not article.title or len(article.title) < 3:
        return None

    return article


def normalize_tags(article: Article) -> Article:
    """规范化标签"""
    if article.tags:
        # 转小写、去重、排序
        tags = list(set(tag.lower().strip() for tag in article.tags if tag.strip()))
        tags.sort()
        article = article.model_copy(update={"tags": tags})

    return article


def truncate_description(max_length: int = 300) -> Callable[[Article], Article]:
    """创建描述截断步骤"""

    def step(article: Article) -> Article:
        if article.description and len(article.description) > max_length:
            truncated = article.description[: max_length - 3] + "..."
            article = article.model_copy(update={"description": truncated})
        return article

    return step


def filter_by_tags(allowed_tags: list[str]) -> Callable[[Article], Article | None]:
    """创建标签过滤步骤"""
    allowed = set(tag.lower() for tag in allowed_tags)

    def step(article: Article) -> Article | None:
        if not article.tags:
            return article  # 没有标签的文章通过

        article_tags = set(tag.lower() for tag in article.tags)
        if article_tags & allowed:
            return article

        return None  # 没有匹配的标签，丢弃

    return step


def create_default_pipeline() -> Pipeline:
    """
    创建默认管道

    包含常用的处理步骤
    """
    pipeline = Pipeline()
    pipeline.add_step(validate_url)
    pipeline.add_step(validate_title)
    pipeline.add_step(clean_html)
    pipeline.add_step(normalize_tags)
    pipeline.add_step(truncate_description(500))
    return pipeline

