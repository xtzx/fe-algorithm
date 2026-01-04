"""
报告生成器

支持:
- Markdown 报告
- JSON 报告
- 终端输出
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blog_aggregator.models import AggregateStats, Article


class Reporter:
    """
    报告生成器

    Example:
        ```python
        reporter = Reporter(articles)

        # 生成 Markdown
        md = reporter.to_markdown()

        # 生成 JSON
        data = reporter.to_json()

        # 保存报告
        reporter.save("report.md", format="markdown")
        ```
    """

    def __init__(
        self,
        articles: list[Article],
        stats: AggregateStats | None = None,
    ) -> None:
        self.articles = articles
        self.stats = stats

    def to_markdown(
        self,
        title: str = "技术博客聚合报告",
        max_articles: int = 50,
        group_by_source: bool = True,
    ) -> str:
        """生成 Markdown 报告"""
        lines = []

        # 标题
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 统计概览
        if self.stats:
            lines.append("## 📊 采集统计")
            lines.append("")
            lines.append(f"- 总源数: {self.stats.total_sources}")
            lines.append(f"- 成功: {self.stats.successful_sources}")
            lines.append(f"- 失败: {self.stats.failed_sources}")
            lines.append(f"- 采集文章: {self.stats.total_articles}")
            lines.append(f"- 新文章: {self.stats.new_articles}")
            lines.append(f"- 耗时: {self.stats.total_elapsed:.1f}秒")
            lines.append("")

        # 文章统计
        lines.append("## 📈 文章统计")
        lines.append("")
        lines.append(f"- 总文章数: {len(self.articles)}")

        # 按源统计
        by_source: dict[str, int] = {}
        for article in self.articles:
            by_source[article.source] = by_source.get(article.source, 0) + 1

        lines.append("- 按来源分布:")
        for source, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {source}: {count}")
        lines.append("")

        # 热门标签
        by_tag: dict[str, int] = {}
        for article in self.articles:
            for tag in article.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        if by_tag:
            top_tags = sorted(by_tag.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("## 🏷️ 热门标签")
            lines.append("")
            for tag, count in top_tags:
                lines.append(f"- `{tag}`: {count}")
            lines.append("")

        # 文章列表
        lines.append("## 📝 文章列表")
        lines.append("")

        if group_by_source:
            # 按源分组
            for source in sorted(by_source.keys()):
                source_articles = [
                    a for a in self.articles if a.source == source
                ][:max_articles]

                lines.append(f"### {source.replace('_', ' ').title()}")
                lines.append("")

                for article in source_articles:
                    self._format_article(article, lines)

                lines.append("")
        else:
            # 按时间排序
            sorted_articles = sorted(
                self.articles,
                key=lambda a: a.published_at or a.collected_at,
                reverse=True,
            )[:max_articles]

            for article in sorted_articles:
                self._format_article(article, lines)

        return "\n".join(lines)

    def _format_article(self, article: Article, lines: list[str]) -> None:
        """格式化单篇文章"""
        lines.append(f"#### [{article.title}]({article.url})")
        lines.append("")

        if article.author:
            lines.append(f"**作者**: {article.author}")

        if article.published_at:
            lines.append(
                f"**发布时间**: {article.published_at.strftime('%Y-%m-%d')}"
            )

        if article.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in article.tags[:5])
            lines.append(f"**标签**: {tags_str}")

        if article.description:
            lines.append(f"\n> {article.description[:200]}...")

        # 统计
        stats_parts = []
        if article.reactions:
            stats_parts.append(f"👍 {article.reactions}")
        if article.comments:
            stats_parts.append(f"💬 {article.comments}")
        if article.reading_time:
            stats_parts.append(f"⏱️ {article.reading_time}分钟")

        if stats_parts:
            lines.append(f"\n{' | '.join(stats_parts)}")

        lines.append("")

    def to_json(self) -> dict[str, Any]:
        """生成 JSON 报告"""
        return {
            "generated_at": datetime.now().isoformat(),
            "stats": self.stats.summary() if self.stats else None,
            "total_articles": len(self.articles),
            "articles": [a.to_dict() for a in self.articles],
        }

    def to_terminal(self, max_articles: int = 20) -> str:
        """生成终端输出"""
        lines = []

        lines.append("=" * 60)
        lines.append("  技术博客聚合报告")
        lines.append("=" * 60)
        lines.append("")

        # 统计
        if self.stats:
            lines.append(f"采集统计:")
            lines.append(f"  源: {self.stats.successful_sources}/{self.stats.total_sources}")
            lines.append(f"  文章: {self.stats.total_articles} (新: {self.stats.new_articles})")
            lines.append(f"  耗时: {self.stats.total_elapsed:.1f}秒")
            lines.append("")

        lines.append(f"总文章数: {len(self.articles)}")
        lines.append("")

        # 文章列表
        lines.append("最新文章:")
        lines.append("-" * 60)

        sorted_articles = sorted(
            self.articles,
            key=lambda a: a.published_at or a.collected_at,
            reverse=True,
        )[:max_articles]

        for i, article in enumerate(sorted_articles, 1):
            title = article.title[:50] + "..." if len(article.title) > 50 else article.title
            source = f"[{article.source}]"
            lines.append(f"{i:2}. {source:12} {title}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        format: str = "markdown",
    ) -> None:
        """
        保存报告

        Args:
            path: 输出路径
            format: 格式（markdown, json）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.to_markdown()
            path.write_text(content, encoding="utf-8")
        elif format == "json":
            data = self.to_json()
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")

