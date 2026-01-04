"""
数据模型

使用 pydantic 进行数据验证
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Article(BaseModel):
    """文章数据模型"""

    # 唯一标识
    id: str = Field(..., description="文章唯一 ID")

    # 基本信息
    title: str = Field(..., min_length=1, description="文章标题")
    url: str = Field(..., description="文章 URL")
    source: str = Field(..., description="来源标识（如 dev_to, hashnode）")

    # 作者信息
    author: str = Field(default="", description="作者名称")
    author_url: str = Field(default="", description="作者主页")

    # 内容摘要
    description: str = Field(default="", description="文章摘要")
    cover_image: str = Field(default="", description="封面图片 URL")

    # 分类
    tags: list[str] = Field(default_factory=list, description="标签列表")

    # 统计
    reading_time: int = Field(default=0, ge=0, description="阅读时间（分钟）")
    reactions: int = Field(default=0, ge=0, description="反应/点赞数")
    comments: int = Field(default=0, ge=0, description="评论数")

    # 时间戳
    published_at: datetime | None = Field(default=None, description="发布时间")
    collected_at: datetime = Field(
        default_factory=datetime.now,
        description="采集时间",
    )

    # 原始数据（用于调试）
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="原始数据",
        exclude=True,
    )

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        """清理标题"""
        return " ".join(v.split()).strip()

    @field_validator("description")
    @classmethod
    def clean_description(cls, v: str) -> str:
        """清理描述"""
        return " ".join(v.split()).strip()[:500]

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: list[str]) -> list[str]:
        """清理标签"""
        return [tag.lower().strip() for tag in v if tag.strip()]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        data = self.model_dump()
        if self.published_at:
            data["published_at"] = self.published_at.isoformat()
        data["collected_at"] = self.collected_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        """从字典创建"""
        if isinstance(data.get("published_at"), str):
            data["published_at"] = datetime.fromisoformat(data["published_at"])
        if isinstance(data.get("collected_at"), str):
            data["collected_at"] = datetime.fromisoformat(data["collected_at"])
        return cls(**data)


class ArticleList(BaseModel):
    """文章列表"""

    articles: list[Article] = Field(default_factory=list)
    source: str = Field(default="", description="来源")
    collected_at: datetime = Field(default_factory=datetime.now)
    total_count: int = Field(default=0, description="总数")

    def __len__(self) -> int:
        return len(self.articles)

    def __iter__(self):
        return iter(self.articles)

    def add(self, article: Article) -> None:
        """添加文章"""
        self.articles.append(article)
        self.total_count = len(self.articles)


class CollectResult(BaseModel):
    """采集结果"""

    source: str = Field(..., description="来源")
    success: bool = Field(default=True, description="是否成功")
    articles_count: int = Field(default=0, description="采集文章数")
    new_count: int = Field(default=0, description="新文章数")
    error: str | None = Field(default=None, description="错误信息")
    elapsed: float = Field(default=0.0, description="耗时（秒）")
    collected_at: datetime = Field(default_factory=datetime.now)


class SourceConfig(BaseModel):
    """源配置"""

    enabled: bool = Field(default=True, description="是否启用")
    base_url: str = Field(..., description="基础 URL")
    api_url: str = Field(default="", description="API URL")
    per_page: int = Field(default=30, ge=1, le=100, description="每页数量")
    per_host_limit: int = Field(default=3, ge=1, le=10, description="每站点并发限制")
    tags: list[str] = Field(default_factory=list, description="要抓取的标签")


class AppConfig(BaseModel):
    """应用配置"""

    data_dir: str = Field(default="data", description="数据目录")
    max_concurrent: int = Field(default=10, ge=1, description="最大并发数")
    rate_limit: float = Field(default=2.0, gt=0, description="每秒请求数")
    timeout: float = Field(default=30.0, gt=0, description="请求超时")
    user_agent: str = Field(
        default="BlogAggregator/1.0",
        description="User-Agent",
    )


class AggregateStats(BaseModel):
    """聚合统计"""

    total_sources: int = Field(default=0)
    successful_sources: int = Field(default=0)
    failed_sources: int = Field(default=0)
    total_articles: int = Field(default=0)
    new_articles: int = Field(default=0)
    total_elapsed: float = Field(default=0.0)

    results: list[CollectResult] = Field(default_factory=list)

    def add_result(self, result: CollectResult) -> None:
        """添加结果"""
        self.results.append(result)
        self.total_sources += 1
        if result.success:
            self.successful_sources += 1
            self.total_articles += result.articles_count
            self.new_articles += result.new_count
        else:
            self.failed_sources += 1
        self.total_elapsed += result.elapsed

    def summary(self) -> dict[str, Any]:
        """返回摘要"""
        return {
            "total_sources": self.total_sources,
            "successful_sources": self.successful_sources,
            "failed_sources": self.failed_sources,
            "total_articles": self.total_articles,
            "new_articles": self.new_articles,
            "total_elapsed_seconds": round(self.total_elapsed, 2),
        }

