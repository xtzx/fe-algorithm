"""测试配置"""

import pytest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_html():
    """简单 HTML 样本"""
    return (FIXTURES_DIR / "simple.html").read_text()


@pytest.fixture
def article_html():
    """文章 HTML 样本"""
    return (FIXTURES_DIR / "article.html").read_text()


@pytest.fixture
def base_url():
    """基础 URL"""
    return "https://example.com"


@pytest.fixture
def robots_txt():
    """robots.txt 样本"""
    return """
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /admin/public/
Crawl-delay: 10

User-agent: BadBot
Disallow: /

Sitemap: https://example.com/sitemap.xml
"""

