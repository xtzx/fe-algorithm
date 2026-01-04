"""解析模块测试"""

import pytest

from scraper import HtmlParser, extract_links, extract_text
from scraper.parser import extract_meta, filter_links, is_valid_url, normalize_url


class TestHtmlParser:
    """HTML 解析器测试"""

    def test_parse(self, simple_html):
        parser = HtmlParser()
        soup = parser.parse(simple_html)
        assert soup is not None

    def test_extract_text(self, simple_html):
        parser = HtmlParser()
        soup = parser.parse(simple_html)

        title = parser.extract_text(soup, "title")
        assert title == "Simple Test Page"

        h1 = parser.extract_text(soup, "h1")
        assert h1 == "Welcome to Simple Page"

    def test_extract_links(self, simple_html, base_url):
        parser = HtmlParser()
        soup = parser.parse(simple_html)

        links = parser.extract_links(soup, base_url)

        assert len(links) == 3
        assert f"{base_url}/page1" in links
        assert f"{base_url}/page2" in links


class TestExtractText:
    """extract_text 函数测试"""

    def test_extract_title(self, article_html):
        text = extract_text(article_html, "h1.title")
        assert text == "Python Web Scraping Guide"

    def test_extract_content(self, article_html):
        text = extract_text(article_html, "article.content")
        assert "Web scraping" in text
        assert "Python provides" in text


class TestExtractLinks:
    """extract_links 函数测试"""

    def test_extract_all(self, article_html, base_url):
        links = extract_links(article_html, base_url)
        assert len(links) > 0
        assert any("/article/" in link for link in links)

    def test_absolute_urls(self, simple_html, base_url):
        links = extract_links(simple_html, base_url)
        # 所有链接都应该是绝对 URL
        assert all(link.startswith("http") for link in links)


class TestExtractMeta:
    """extract_meta 函数测试"""

    def test_basic_meta(self, article_html):
        meta = extract_meta(article_html)

        assert meta["title"] == "Python Web Scraping Guide"
        assert "scraping" in meta["description"]
        assert "python" in meta["keywords"]

    def test_og_meta(self, article_html):
        meta = extract_meta(article_html)

        assert meta["og_title"] == "OG Title"
        assert meta["og_description"] == "OG Description"


class TestNormalizeUrl:
    """normalize_url 函数测试"""

    def test_remove_fragment(self):
        url = normalize_url("https://example.com/page#section")
        assert url == "https://example.com/page"

    def test_remove_trailing_slash(self):
        url = normalize_url("https://example.com/page/")
        assert url == "https://example.com/page"

    def test_keep_root_slash(self):
        url = normalize_url("https://example.com/")
        assert url == "https://example.com/"


class TestIsValidUrl:
    """is_valid_url 函数测试"""

    def test_valid_urls(self):
        assert is_valid_url("https://example.com") is True
        assert is_valid_url("http://example.com/page") is True

    def test_invalid_urls(self):
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("javascript:void(0)") is False
        assert is_valid_url("") is False

    def test_allowed_domains(self):
        assert is_valid_url(
            "https://example.com/page",
            allowed_domains=["example.com"],
        ) is True
        assert is_valid_url(
            "https://other.com/page",
            allowed_domains=["example.com"],
        ) is False


class TestFilterLinks:
    """filter_links 函数测试"""

    def test_same_domain(self, article_html, base_url):
        links = extract_links(article_html, base_url)
        filtered = filter_links(links, base_url, same_domain=True)

        # 应该只保留同域名链接
        for link in filtered:
            assert "example.com" in link

    def test_exclude_patterns(self, simple_html, base_url):
        links = extract_links(simple_html, base_url)
        filtered = filter_links(
            links,
            base_url,
            same_domain=False,
            exclude_patterns=[r"/page1"],
        )

        assert not any("/page1" in link for link in filtered)

