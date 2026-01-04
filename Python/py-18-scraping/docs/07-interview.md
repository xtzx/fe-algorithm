# 面试题

## 1. 如何遵守 robots.txt？

**答案**：

**步骤**：
1. 获取 robots.txt 文件
2. 解析 User-agent 和 Disallow 规则
3. 检查 URL 是否被允许

```python
from scraper import RobotsChecker

checker = RobotsChecker(user_agent="MyBot")

# 检查 URL
if await checker.is_allowed("https://example.com/page"):
    # 可以爬取
    pass

# 获取建议延迟
delay = await checker.get_crawl_delay("https://example.com")
```

**关键点**：
- 每个域名只需获取一次
- 缓存解析结果
- 遵守 Crawl-delay

---

## 2. 如何处理反爬机制？

**答案**：

**合规方式**：

1. **降低请求频率**
```python
fetcher = RateLimitedFetcher(requests_per_second=1)
```

2. **设置合理的 User-Agent**
```python
user_agent = "MyBot/1.0 (+https://example.com/bot)"
```

3. **遵守 robots.txt**
```python
if await robots_checker.is_allowed(url):
    await fetcher.fetch(url)
```

4. **处理 429 错误**
```python
if result.status_code == 429:
    retry_after = result.headers.get("Retry-After", 60)
    await asyncio.sleep(float(retry_after))
```

**不推荐的做法**：
- 伪装 User-Agent
- 大规模使用代理
- 自动破解验证码

---

## 3. 如何测试爬虫？

**答案**：

**策略**：

1. **纯函数化解析逻辑**
```python
# 可单独测试
def parse_article(html: str) -> dict:
    ...
```

2. **使用 HTML fixture**
```python
@pytest.fixture
def article_html():
    return Path("fixtures/article.html").read_text()

def test_parse_article(article_html):
    result = parse_article(article_html)
    assert result["title"] == "Expected Title"
```

3. **Mock 网络请求**
```python
@respx.mock
async def test_fetcher():
    respx.get("https://example.com/").respond(200, text="...")
    result = await fetcher.fetch("https://example.com/")
    assert result.success
```

---

## 4. 如何实现断点续爬？

**答案**：

**实现方式**：

1. **保存状态到文件**
```python
state = {
    "pending_urls": [...],
    "processed_urls": [...],
    "failed_urls": [...],
}
json.dump(state, file)
```

2. **启动时恢复**
```python
if state_file.exists():
    state = json.load(state_file)
    # 从 pending_urls 继续
```

3. **定期保存**
```python
# 每处理 N 个 URL 保存一次
if processed_count % 100 == 0:
    save_state()
```

**使用 scraper**：
```python
from scraper import FileState, StateManager

manager = StateManager(FileState("state.json"))
# 自动保存和恢复
```

---

## 5. 如何去重 URL？

**答案**：

**方法**：

1. **HashSet（小规模）**
```python
seen = set()
if url not in seen:
    seen.add(url)
    process(url)
```

2. **布隆过滤器（大规模）**
```python
from scraper import BloomFilter

dedup = BloomFilter(expected_items=1000000)
if dedup.add(url):  # 返回 True 表示新 URL
    process(url)
```

3. **URL 标准化**
```python
def normalize(url):
    # 移除 fragment
    url = url.split("#")[0]
    # 移除尾部斜杠
    return url.rstrip("/")
```

---

## 6. 如何处理 JavaScript 渲染的页面？

**答案**：

**解决方案**：

1. **Playwright（推荐）**
```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    await page.goto(url)
    await page.wait_for_load_state("networkidle")
    html = await page.content()
```

2. **查找 API**
```python
# 分析网络请求，直接调用 API
response = await client.get("https://example.com/api/data")
data = response.json()
```

3. **预渲染服务**
```python
# 使用 Prerender.io 等服务
prerender_url = f"https://prerender.io/?url={url}"
```

**选择建议**：
- 先查找 API
- API 不可用时使用 Playwright
- 大规模时考虑预渲染服务

---

## 7. 如何限制爬取速率？

**答案**：

**方法**：

1. **固定延迟**
```python
for url in urls:
    await fetcher.fetch(url)
    await asyncio.sleep(1.0)  # 1 秒间隔
```

2. **令牌桶**
```python
from scraper import RateLimitedFetcher

fetcher = RateLimitedFetcher(
    requests_per_second=2.0,
    jitter=0.5,  # 随机延迟
)
```

3. **遵守 Crawl-delay**
```python
delay = await robots_checker.get_crawl_delay(url)
if delay:
    await asyncio.sleep(delay)
```

---

## 8. 爬虫的法律风险？

**答案**：

**可能违法的行为**：

1. **绕过访问控制**
   - 破解登录
   - 绕过验证码
   - 利用安全漏洞

2. **侵犯版权**
   - 大量复制受保护内容
   - 商业使用爬取内容

3. **违反服务条款**
   - 很多网站明确禁止爬取

4. **侵犯隐私**
   - 收集个人信息
   - 违反 GDPR

**安全做法**：
- 只爬公开数据
- 遵守 robots.txt
- 控制请求频率
- 阅读服务条款
- 不存储个人信息

---

## 9. 如何设计可扩展的爬虫架构？

**答案**：

**组件分离**：

```
┌─────────────┐
│   调度器     │  URL 分发
└──────┬──────┘
       │
┌──────▼──────┐
│   Fetcher    │  HTTP 请求
└──────┬──────┘
       │
┌──────▼──────┐
│   Parser     │  HTML 解析
└──────┬──────┘
       │
┌──────▼──────┐
│  Pipeline    │  数据处理
└─────────────┘
```

**关键设计**：
- 解析逻辑纯函数化
- 使用队列解耦
- 支持断点续爬
- 可配置的管道

---

## 10. 如何处理爬虫中的错误？

**答案**：

**错误类型和处理**：

| 错误 | 处理方式 |
|------|----------|
| 网络超时 | 重试 |
| 连接失败 | 重试 + 退避 |
| 404 | 记录并跳过 |
| 429 | 等待 Retry-After |
| 500 | 重试几次后放弃 |
| 解析失败 | 记录 HTML 用于调试 |

```python
result = await fetcher.fetch(url)

if result.status_code == 429:
    await handle_rate_limit(result)
elif result.status_code == 404:
    logger.warning(f"Not found: {url}")
elif result.status_code >= 500:
    await retry_queue.put(url)
elif not result.success:
    logger.error(f"Failed: {url} - {result.error}")
```

