# 工程化设计

> URL 管理、去重、断点续爬、失败重试

## 1. URL 管理

### URL 队列

```python
from scraper import UrlQueue, HashSetDedup

queue = UrlQueue(dedup=HashSetDedup())

# 添加 URL
queue.add("https://example.com/page1")
queue.add("https://example.com/page2")
queue.add("https://example.com/page1")  # 重复，不会添加

# 获取下一个
url = queue.pop()
```

### URL 标准化

```python
from scraper.parser import normalize_url

# 标准化 URL
url = normalize_url("https://example.com/page#section")
# -> "https://example.com/page"

url = normalize_url("https://example.com/path/")
# -> "https://example.com/path"
```

## 2. URL 去重

### HashSet 去重（小规模）

```python
from scraper import HashSetDedup

dedup = HashSetDedup()

# 添加并检查
if dedup.add("https://example.com/page"):
    print("New URL!")
else:
    print("Already seen")

# 保存/加载
dedup.save("seen_urls.json")
dedup = HashSetDedup.load("seen_urls.json")
```

### 布隆过滤器（大规模）

```python
from scraper import BloomFilter

# 适合百万级 URL
dedup = BloomFilter(
    expected_items=1000000,
    false_positive_rate=0.01,
)

# 注意：可能有假阳性
dedup.add("https://example.com")
dedup.contains("https://example.com")  # True
```

### 持久化去重

```python
from scraper.dedup import PersistentDedup

# 每次添加都写入文件
dedup = PersistentDedup("seen_urls.txt")
dedup.add("https://example.com")  # 立即持久化
```

## 3. 断点续爬

### 状态管理

```python
from scraper import FileState, StateManager

# 使用文件保存状态
state = FileState("crawl_state.json")
manager = StateManager(state)

# 添加待处理 URL
manager.add_pending("https://example.com/page1")

# 获取下一个
url = manager.next_pending()

# 标记为已处理
manager.mark_processed(url)

# 保存状态
manager.save()
```

### 恢复爬取

```python
# 程序重启后，自动从状态文件恢复
manager = StateManager(FileState("crawl_state.json"))

print(f"Pending: {manager.pending_count}")
print(f"Processed: {manager.processed_count}")

# 继续爬取
while url := manager.next_pending():
    result = await fetcher.fetch(url)
    if result.success:
        manager.mark_processed(url)
    else:
        manager.mark_failed(url)
```

### 自动保存

```python
# FileState 支持自动保存
state = FileState(
    "crawl_state.json",
    auto_save_interval=100,  # 每 100 次操作自动保存
)
```

## 4. 失败重试

### 重试配置

```python
from scraper import Fetcher

fetcher = Fetcher(
    max_retries=3,       # 最大重试次数
    retry_delay=1.0,     # 重试延迟
)
```

### 失败队列

```python
from scraper import StateManager

manager = StateManager(state)

# 标记失败
manager.mark_failed(url)

# 后续重试失败的 URL
retry_count = manager.retry_failed()
print(f"Retrying {retry_count} failed URLs")
```

## 5. 数据持久化

### JSONL 格式

```python
from scraper import Pipeline, JsonLineWriter

pipeline = Pipeline([
    JsonLineWriter("items.jsonl"),
])

async with pipeline:
    await pipeline.process({"url": "...", "title": "..."})
```

### JSONL 格式示例

```jsonl
{"url": "https://example.com/1", "title": "Page 1"}
{"url": "https://example.com/2", "title": "Page 2"}
{"url": "https://example.com/3", "title": "Page 3"}
```

### 读取 JSONL

```python
from scraper.pipeline import load_jsonl

items = load_jsonl("items.jsonl")
for item in items:
    print(item["title"])
```

## 6. 数据管道

### 清洗步骤

```python
from scraper.pipeline import CleanStep, ValidateStep, Pipeline

pipeline = Pipeline([
    CleanStep(strip_strings=True, remove_empty=True),
    ValidateStep(required_fields=["url", "title"]),
    JsonLineWriter("items.jsonl"),
])
```

### 去重步骤

```python
from scraper.pipeline import DeduplicateStep

pipeline = Pipeline([
    DeduplicateStep(field="url"),
    JsonLineWriter("items.jsonl"),
])
```

## 7. 完整示例

```python
import asyncio
from scraper import (
    RateLimitedFetcher,
    HtmlParser,
    UrlQueue,
    HashSetDedup,
    FileState,
    StateManager,
    Pipeline,
    JsonLineWriter,
    RobotsChecker,
    extract_links,
    filter_links,
)

async def crawl(start_url: str, max_pages: int = 100):
    # 初始化组件
    queue = UrlQueue(HashSetDedup())
    manager = StateManager(FileState("state.json"))
    parser = HtmlParser()
    robots = RobotsChecker()

    pipeline = Pipeline([
        JsonLineWriter("items.jsonl"),
    ])

    queue.add(start_url)
    manager.add_pending(start_url)
    pages = 0

    async with RateLimitedFetcher(requests_per_second=2) as fetcher:
        async with pipeline:
            while not queue.is_empty and pages < max_pages:
                url = queue.pop()

                # 检查 robots.txt
                if not await robots.is_allowed(url):
                    continue

                # 获取页面
                result = await fetcher.fetch(url)
                if not result.success:
                    manager.mark_failed(url)
                    continue

                manager.mark_processed(url)
                pages += 1

                # 解析
                soup = parser.parse(result.html)
                title = parser.extract_text(soup, "title")

                # 保存
                await pipeline.process({
                    "url": url,
                    "title": title,
                })

                # 提取链接
                links = extract_links(result.html, url)
                links = filter_links(links, start_url, same_domain=True)
                queue.add_many(links)

    manager.save()
    print(f"Crawled {pages} pages")

asyncio.run(crawl("https://example.com"))
```

## 小结

| 功能 | 组件 |
|------|------|
| URL 队列 | UrlQueue |
| URL 去重 | HashSetDedup / BloomFilter |
| 断点续爬 | FileState + StateManager |
| 失败重试 | Fetcher.max_retries |
| 数据持久化 | Pipeline + JsonLineWriter |

