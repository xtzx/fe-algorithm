# 爬虫基础

> 静态页面抓取、HTML 解析、链接提取

## 1. HTTP 请求

### 使用 httpx

```python
import httpx

async def fetch_page(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
```

### 设置 User-Agent

```python
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; MyBot/1.0; +https://example.com/bot)",
}

async with httpx.AsyncClient(headers=headers) as client:
    response = await client.get(url)
```

## 2. HTML 解析

### BeautifulSoup 基础

```python
from bs4 import BeautifulSoup

html = "<html><body><h1>Hello</h1></body></html>"
soup = BeautifulSoup(html, "lxml")

# 选择元素
h1 = soup.select_one("h1")
print(h1.text)  # "Hello"
```

### CSS 选择器

```python
# 单个元素
soup.select_one("h1")
soup.select_one("div.class-name")
soup.select_one("div#id-name")
soup.select_one("a[href]")

# 多个元素
soup.select("a")
soup.select("div.item")
soup.select("ul > li")
```

### 提取数据

```python
# 文本
text = element.get_text()
text = element.get_text(strip=True)

# 属性
href = element.get("href")
src = element["src"]

# 遍历
for link in soup.select("a[href]"):
    href = link.get("href")
    text = link.get_text()
```

## 3. 链接提取

### 基础提取

```python
from urllib.parse import urljoin

base_url = "https://example.com/page/"

links = []
for a in soup.select("a[href]"):
    href = a.get("href")
    # 转换为绝对 URL
    absolute_url = urljoin(base_url, href)
    links.append(absolute_url)
```

### 使用 scraper 库

```python
from scraper import extract_links, filter_links

# 提取所有链接
links = extract_links(html, base_url)

# 过滤链接
links = filter_links(
    links,
    base_url,
    same_domain=True,
    exclude_patterns=[r"/admin/", r"\.pdf$"],
)
```

## 4. 动态页面概念

### 什么是动态页面

- 使用 JavaScript 渲染内容
- 数据通过 AJAX 加载
- 需要浏览器执行 JS

### 解决方案

1. **Playwright** - 无头浏览器
2. **查找 API** - 直接请求数据接口
3. **预渲染服务** - 如 Prerender.io

### Playwright 示例

```python
from playwright.async_api import async_playwright

async def fetch_dynamic(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        html = await page.content()
        await browser.close()
        return html
```

## 5. 使用 scraper 库

### 基础爬取

```python
from scraper import Fetcher, HtmlParser

async def main():
    parser = HtmlParser()

    async with Fetcher() as fetcher:
        result = await fetcher.fetch("https://example.com")

        if result.success:
            soup = parser.parse(result.html)
            title = parser.extract_text(soup, "title")
            print(f"Title: {title}")
```

### 带速率限制

```python
from scraper import RateLimitedFetcher

async with RateLimitedFetcher(requests_per_second=2) as fetcher:
    for url in urls:
        result = await fetcher.fetch(url)
        # 自动限制请求频率
```

## 6. 与 JS 对比

```python
# Python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "lxml")
title = soup.select_one("h1").text
```

```javascript
// JavaScript with cheerio
const cheerio = require("cheerio");

const $ = cheerio.load(html);
const title = $("h1").text();
```

## 小结

| 概念 | 工具 |
|------|------|
| HTTP 请求 | httpx |
| HTML 解析 | BeautifulSoup |
| CSS 选择器 | soup.select() |
| 动态页面 | Playwright |

