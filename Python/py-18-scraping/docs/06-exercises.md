# 练习题

## 练习 1：基础页面抓取（⭐）

使用 httpx 和 BeautifulSoup 抓取页面标题：

```python
async def fetch_title(url: str) -> str:
    """
    获取页面标题

    Example:
        title = await fetch_title("https://example.com")
        # -> "Example Domain"
    """
    pass
```

---

## 练习 2：提取所有链接（⭐）

从页面提取所有链接：

```python
def extract_all_links(html: str, base_url: str) -> list[str]:
    """
    提取页面中的所有链接

    要求:
    1. 将相对链接转为绝对链接
    2. 过滤非 HTTP(S) 链接
    """
    pass
```

---

## 练习 3：解析 robots.txt（⭐）

实现 robots.txt 检查：

```python
def is_allowed_by_robots(
    robots_content: str,
    path: str,
    user_agent: str = "*",
) -> bool:
    """
    检查路径是否被 robots.txt 允许
    """
    pass

# 测试
robots = """
User-agent: *
Disallow: /admin/
"""
assert is_allowed_by_robots(robots, "/page") == True
assert is_allowed_by_robots(robots, "/admin/") == False
```

---

## 练习 4：URL 去重（⭐⭐）

实现 URL 去重器：

```python
class UrlDeduplicator:
    """
    URL 去重器

    要求:
    1. 标准化 URL（移除 fragment、尾部斜杠）
    2. 支持 add() 和 contains()
    """

    def add(self, url: str) -> bool:
        """添加 URL，返回是否是新 URL"""
        pass

    def contains(self, url: str) -> bool:
        """检查 URL 是否已存在"""
        pass
```

---

## 练习 5：请求限流（⭐⭐）

实现请求限流器：

```python
class RateLimiter:
    """
    请求限流器

    使用令牌桶算法
    """

    def __init__(self, requests_per_second: float):
        pass

    async def acquire(self):
        """获取许可，必要时等待"""
        pass
```

---

## 练习 6：断点续爬（⭐⭐）

实现可恢复的爬虫状态：

```python
class CrawlerState:
    """
    爬虫状态管理

    支持:
    1. 保存/加载状态到文件
    2. 记录已处理和待处理的 URL
    """

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> "CrawlerState":
        pass
```

---

## 练习 7：JSONL 输出（⭐⭐）

实现 JSONL 写入器：

```python
class JsonLineWriter:
    """
    JSONL 写入器

    每行一个 JSON 对象
    """

    def __init__(self, path: str):
        pass

    def write(self, item: dict):
        pass

    def close(self):
        pass
```

---

## 练习 8：文章解析器（⭐⭐⭐）

创建可配置的文章解析器：

```python
def create_article_parser(
    title_selector: str,
    content_selector: str,
    date_selector: str | None = None,
):
    """
    创建文章解析器（纯函数工厂）

    返回一个解析函数
    """
    def parse(html: str, url: str) -> dict:
        pass

    return parse
```

---

## 练习 9：测试解析逻辑（⭐⭐⭐）

为解析函数编写测试：

```python
# 给定以下解析函数
def parse_product(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    return {
        "name": soup.select_one("h1.product-name").text.strip(),
        "price": soup.select_one("span.price").text.strip(),
    }

# 编写测试
def test_parse_product():
    """
    使用 HTML fixture 测试解析函数
    """
    pass
```

---

## 练习 10：数据管道（⭐⭐⭐）

实现数据处理管道：

```python
class Pipeline:
    """
    数据处理管道

    支持链式处理:
    1. 清洗
    2. 验证
    3. 存储
    """

    def __init__(self, steps: list):
        pass

    async def process(self, item: dict) -> dict | None:
        """处理数据项，返回 None 表示丢弃"""
        pass
```

---

## 练习 11：Mock 网络测试（⭐⭐⭐）

使用 respx 测试网络请求：

```python
@respx.mock
async def test_fetcher():
    """
    测试 Fetcher 类

    要求:
    1. Mock 成功响应
    2. Mock 失败响应
    3. 测试重试逻辑
    """
    pass
```

---

## 练习 12：完整爬虫（⭐⭐⭐⭐）

实现一个完整的爬虫：

```python
class SimpleCrawler:
    """
    简单爬虫

    功能:
    1. 遵守 robots.txt
    2. 请求限流
    3. URL 去重
    4. 断点续爬
    5. JSONL 输出
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = 100,
        delay: float = 1.0,
    ):
        pass

    async def crawl(self):
        """开始爬取"""
        pass
```

---

## 练习答案提示

1. `httpx.get()` + `BeautifulSoup.select_one("title")`
2. `urljoin()` 转换相对链接
3. 解析 User-agent 和 Disallow 指令
4. 使用 Set 存储，标准化 URL
5. 令牌桶或时间间隔
6. JSON 序列化状态到文件
7. `json.dumps()` + 换行
8. 闭包捕获选择器配置
9. 准备 HTML fixture 文件
10. 依次调用每个步骤
11. `respx.get().respond()` 设置 mock
12. 组合以上所有组件

