# 可测试设计

> 解析纯函数化、fixture 测试、mock 网络请求

## 1. 解析逻辑纯函数化

### 为什么要纯函数

- 易于测试
- 无副作用
- 可重用

### 示例

```python
# ✅ 推荐：纯函数
def parse_article(html: str, url: str) -> dict:
    """
    纯函数：输入 HTML，输出数据
    不依赖网络，不依赖状态
    """
    soup = BeautifulSoup(html, "lxml")
    return {
        "url": url,
        "title": soup.select_one("h1").text.strip(),
        "content": soup.select_one("article").text.strip(),
    }

# ❌ 不推荐：混合网络和解析
async def get_article(url: str) -> dict:
    """
    混合了网络请求和解析
    难以单独测试解析逻辑
    """
    response = await httpx.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    return {"title": soup.select_one("h1").text}
```

### scraper 中的纯函数

```python
from scraper import extract_text, extract_links, extract_meta

# 这些都是纯函数
text = extract_text(html, "article")
links = extract_links(html, base_url)
meta = extract_meta(html)
```

## 2. Fixture 测试

### 准备 HTML 样本

```html
<!-- tests/fixtures/article.html -->
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
  <h1>Article Title</h1>
  <article>
    <p>Article content here.</p>
  </article>
  <a href="/page1">Link 1</a>
  <a href="/page2">Link 2</a>
</body>
</html>
```

### 使用 pytest fixture

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def article_html():
    """加载 HTML 样本"""
    path = Path(__file__).parent / "fixtures/article.html"
    return path.read_text()

@pytest.fixture
def base_url():
    return "https://example.com"
```

### 测试解析函数

```python
# tests/test_parser.py
from scraper import extract_text, extract_links

def test_extract_title(article_html):
    text = extract_text(article_html, "h1")
    assert text == "Article Title"

def test_extract_links(article_html, base_url):
    links = extract_links(article_html, base_url)
    assert len(links) == 2
    assert "https://example.com/page1" in links
```

## 3. Mock 网络请求

### 使用 respx

```python
import pytest
import respx
import httpx

@respx.mock
async def test_fetch_page():
    # 设置 mock
    respx.get("https://example.com/").respond(
        status_code=200,
        text="<html><body>Hello</body></html>",
    )

    # 测试
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com/")
        assert response.status_code == 200
        assert "Hello" in response.text
```

### 测试 Fetcher

```python
import respx
from scraper import Fetcher

@respx.mock
async def test_fetcher_success():
    respx.get("https://example.com/").respond(
        status_code=200,
        text="<html><body>Test</body></html>",
    )

    async with Fetcher() as fetcher:
        result = await fetcher.fetch("https://example.com/")

    assert result.success
    assert result.status_code == 200
    assert "Test" in result.html

@respx.mock
async def test_fetcher_retry():
    # 第一次失败，第二次成功
    respx.get("https://example.com/").mock(
        side_effect=[
            httpx.ConnectError("Connection failed"),
            httpx.Response(200, text="Success"),
        ]
    )

    async with Fetcher(max_retries=3) as fetcher:
        result = await fetcher.fetch("https://example.com/")

    assert result.success
```

## 4. 测试去重

```python
from scraper import HashSetDedup

def test_dedup_basic():
    dedup = HashSetDedup()

    # 第一次添加返回 True
    assert dedup.add("https://example.com/page1") is True

    # 重复添加返回 False
    assert dedup.add("https://example.com/page1") is False

    # 不同 URL 返回 True
    assert dedup.add("https://example.com/page2") is True

def test_dedup_normalization():
    dedup = HashSetDedup()

    # URL 应该被标准化
    dedup.add("https://example.com/page#section")
    assert dedup.contains("https://example.com/page") is True
```

## 5. 测试状态管理

```python
import pytest
from pathlib import Path
from scraper import FileState, StateManager

@pytest.fixture
def temp_state_file(tmp_path):
    return tmp_path / "state.json"

def test_state_persistence(temp_state_file):
    # 创建状态
    manager = StateManager(FileState(temp_state_file))
    manager.add_pending("https://example.com/page1")
    manager.mark_processed("https://example.com/page1")
    manager.save()

    # 重新加载
    manager2 = StateManager(FileState(temp_state_file))
    assert manager2.is_processed("https://example.com/page1")
    assert manager2.processed_count == 1
```

## 6. 测试数据管道

```python
import pytest
from scraper import Pipeline, JsonLineWriter, CleanStep
from scraper.pipeline import load_jsonl

@pytest.fixture
def temp_output(tmp_path):
    return tmp_path / "items.jsonl"

async def test_pipeline(temp_output):
    pipeline = Pipeline([
        CleanStep(),
        JsonLineWriter(temp_output),
    ])

    async with pipeline:
        await pipeline.process({"url": "https://example.com", "title": "Test"})

    items = load_jsonl(temp_output)
    assert len(items) == 1
    assert items[0]["title"] == "Test"
```

## 7. 测试 robots.txt

```python
from scraper.robots import RobotsParser, check_robots_txt

def test_robots_parser():
    robots_txt = """
User-agent: *
Disallow: /admin/
Allow: /admin/public/
Crawl-delay: 10
"""

    parser = RobotsParser()
    parser.parse(robots_txt)

    assert parser.is_allowed("/page") is True
    assert parser.is_allowed("/admin/") is False
    assert parser.is_allowed("/admin/public/") is True
    assert parser.get_crawl_delay() == 10.0

def test_check_robots_txt():
    robots_txt = "User-agent: *\nDisallow: /private/"

    assert check_robots_txt(robots_txt, "/page") is True
    assert check_robots_txt(robots_txt, "/private/") is False
```

## 8. 测试组织

```
tests/
├── conftest.py          # 共享 fixture
├── fixtures/            # HTML 样本
│   ├── simple.html
│   ├── article.html
│   └── robots.txt
├── test_fetcher.py      # 网络请求测试
├── test_parser.py       # 解析测试
├── test_dedup.py        # 去重测试
├── test_state.py        # 状态管理测试
├── test_pipeline.py     # 管道测试
└── test_robots.py       # robots.txt 测试
```

## 小结

| 原则 | 做法 |
|------|------|
| 纯函数化 | 解析逻辑与网络分离 |
| Fixture | 使用 HTML 样本文件 |
| Mock | 使用 respx mock 网络 |
| 隔离 | 测试各模块独立 |

