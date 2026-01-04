"""
页面解析模块

支持:
- HTML 解析
- CSS 选择器
- 链接提取
- 文本提取
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag


@dataclass
class ParsedItem:
    """解析结果"""

    url: str
    title: str = ""
    content: str = ""
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HtmlParser:
    """
    HTML 解析器

    Example:
        ```python
        parser = HtmlParser()
        soup = parser.parse(html)

        # CSS 选择器
        titles = parser.select_all(soup, "h1.title")

        # 提取文本
        text = parser.extract_text(soup, "article")
        ```
    """

    def __init__(self, parser: str = "lxml") -> None:
        """
        初始化解析器

        Args:
            parser: BeautifulSoup 解析器 ("lxml", "html.parser", "html5lib")
        """
        self.parser = parser

    def parse(self, html: str) -> BeautifulSoup:
        """解析 HTML"""
        return BeautifulSoup(html, self.parser)

    def select(self, soup: BeautifulSoup | Tag, selector: str) -> Tag | None:
        """选择单个元素"""
        return soup.select_one(selector)

    def select_all(self, soup: BeautifulSoup | Tag, selector: str) -> list[Tag]:
        """选择所有匹配元素"""
        return soup.select(selector)

    def extract_text(
        self,
        soup: BeautifulSoup | Tag,
        selector: str | None = None,
        strip: bool = True,
    ) -> str:
        """
        提取文本内容

        Args:
            soup: BeautifulSoup 或 Tag 对象
            selector: CSS 选择器（可选）
            strip: 是否去除空白
        """
        if selector:
            element = self.select(soup, selector)
            if element is None:
                return ""
            soup = element

        text = soup.get_text()
        if strip:
            text = " ".join(text.split())
        return text

    def extract_attr(
        self,
        soup: BeautifulSoup | Tag,
        selector: str,
        attr: str,
    ) -> str | None:
        """提取元素属性"""
        element = self.select(soup, selector)
        if element is None:
            return None
        return element.get(attr)

    def extract_links(
        self,
        soup: BeautifulSoup | Tag,
        base_url: str,
        selector: str = "a[href]",
    ) -> list[str]:
        """
        提取链接

        Args:
            soup: BeautifulSoup 或 Tag 对象
            base_url: 基础 URL（用于相对链接）
            selector: 链接选择器
        """
        links = []
        for a in self.select_all(soup, selector):
            href = a.get("href")
            if href:
                # 转换为绝对 URL
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        return links


# =============================================================================
# 纯函数解析工具
# =============================================================================


def extract_text(html: str, selector: str | None = None) -> str:
    """
    从 HTML 提取文本（纯函数）

    Example:
        ```python
        text = extract_text(html, "article.content")
        ```
    """
    soup = BeautifulSoup(html, "lxml")
    if selector:
        element = soup.select_one(selector)
        if element is None:
            return ""
        return " ".join(element.get_text().split())
    return " ".join(soup.get_text().split())


def extract_links(html: str, base_url: str) -> list[str]:
    """
    从 HTML 提取链接（纯函数）

    Example:
        ```python
        links = extract_links(html, "https://example.com")
        ```
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if href:
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
    return links


def extract_meta(html: str) -> dict[str, str]:
    """
    提取 meta 标签信息

    Returns:
        {
            "title": "...",
            "description": "...",
            "keywords": "...",
        }
    """
    soup = BeautifulSoup(html, "lxml")
    meta = {}

    # 标题
    title_tag = soup.select_one("title")
    if title_tag:
        meta["title"] = title_tag.get_text().strip()

    # meta description
    desc = soup.select_one('meta[name="description"]')
    if desc:
        meta["description"] = desc.get("content", "")

    # meta keywords
    keywords = soup.select_one('meta[name="keywords"]')
    if keywords:
        meta["keywords"] = keywords.get("content", "")

    # Open Graph
    og_title = soup.select_one('meta[property="og:title"]')
    if og_title:
        meta["og_title"] = og_title.get("content", "")

    og_desc = soup.select_one('meta[property="og:description"]')
    if og_desc:
        meta["og_description"] = og_desc.get("content", "")

    return meta


def clean_html(html: str, remove_scripts: bool = True, remove_styles: bool = True) -> str:
    """
    清理 HTML

    Args:
        html: HTML 内容
        remove_scripts: 是否移除 script 标签
        remove_styles: 是否移除 style 标签
    """
    soup = BeautifulSoup(html, "lxml")

    if remove_scripts:
        for script in soup.select("script"):
            script.decompose()

    if remove_styles:
        for style in soup.select("style"):
            style.decompose()

    return str(soup)


def normalize_url(url: str) -> str:
    """
    标准化 URL

    - 移除 fragment (#...)
    - 移除尾部斜杠
    - 转小写 scheme 和 host
    """
    parsed = urlparse(url)

    # 移除 fragment
    normalized = parsed._replace(fragment="")

    # 重建 URL
    result = normalized.geturl()

    # 移除尾部斜杠（除非是根路径）
    if result.endswith("/") and parsed.path != "/":
        result = result.rstrip("/")

    return result


def is_valid_url(url: str, allowed_domains: list[str] | None = None) -> bool:
    """
    验证 URL 是否有效

    Args:
        url: 要验证的 URL
        allowed_domains: 允许的域名列表
    """
    try:
        parsed = urlparse(url)

        # 检查 scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # 检查 domain
        if not parsed.netloc:
            return False

        # 检查是否在允许的域名列表中
        if allowed_domains:
            domain = parsed.netloc.lower()
            if not any(domain == d or domain.endswith(f".{d}") for d in allowed_domains):
                return False

        return True
    except Exception:
        return False


def filter_links(
    links: list[str],
    base_url: str,
    same_domain: bool = True,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """
    过滤链接

    Args:
        links: 链接列表
        base_url: 基础 URL
        same_domain: 是否只保留同域名链接
        exclude_patterns: 排除的正则模式
    """
    base_domain = urlparse(base_url).netloc

    filtered = []
    exclude_re = [re.compile(p) for p in (exclude_patterns or [])]

    for link in links:
        # 跳过无效 URL
        if not is_valid_url(link):
            continue

        # 检查域名
        if same_domain:
            link_domain = urlparse(link).netloc
            if link_domain != base_domain:
                continue

        # 检查排除模式
        if any(r.search(link) for r in exclude_re):
            continue

        filtered.append(link)

    return filtered


# =============================================================================
# 自定义解析器工厂
# =============================================================================


def create_article_parser(
    title_selector: str = "h1",
    content_selector: str = "article",
    date_selector: str | None = None,
    author_selector: str | None = None,
) -> Callable[[str, str], ParsedItem]:
    """
    创建文章解析器

    Example:
        ```python
        parse_article = create_article_parser(
            title_selector="h1.post-title",
            content_selector="div.post-content",
        )

        item = parse_article(html, url)
        ```
    """

    def parser(html: str, url: str) -> ParsedItem:
        soup = BeautifulSoup(html, "lxml")

        # 提取标题
        title_elem = soup.select_one(title_selector)
        title = title_elem.get_text().strip() if title_elem else ""

        # 提取内容
        content_elem = soup.select_one(content_selector)
        content = " ".join(content_elem.get_text().split()) if content_elem else ""

        # 提取链接
        links = extract_links(html, url)

        # 元数据
        metadata: dict[str, Any] = {}

        if date_selector:
            date_elem = soup.select_one(date_selector)
            if date_elem:
                metadata["date"] = date_elem.get_text().strip()

        if author_selector:
            author_elem = soup.select_one(author_selector)
            if author_elem:
                metadata["author"] = author_elem.get_text().strip()

        return ParsedItem(
            url=url,
            title=title,
            content=content,
            links=links,
            metadata=metadata,
        )

    return parser

