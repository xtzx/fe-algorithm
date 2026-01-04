# 高级话题

> 代理池、Cookie/Session、分布式爬虫

## 1. 代理池

### 为什么需要代理

- 避免 IP 被封
- 绕过地理限制
- 提高匿名性

### 代理轮换

```python
from scraper.fetcher import ProxyRotator

rotator = ProxyRotator([
    "http://proxy1:8080",
    "http://proxy2:8080",
    "http://proxy3:8080",
])

# 获取下一个代理
proxy = rotator.get_next()

# 标记失败的代理
rotator.mark_failed(proxy)
```

### 使用代理

```python
from scraper import Fetcher

async with Fetcher(proxy="http://proxy:8080") as fetcher:
    result = await fetcher.fetch(url)
```

### 代理轮换爬取

```python
from scraper.fetcher import fetch_with_proxy_rotation

result = await fetch_with_proxy_rotation(
    url="https://example.com",
    proxies=["http://proxy1:8080", "http://proxy2:8080"],
    max_retries=3,
)
```

## 2. Cookie/Session 管理

### httpx 的 Cookie 管理

```python
import httpx

async with httpx.AsyncClient() as client:
    # 登录
    response = await client.post(
        "https://example.com/login",
        data={"username": "user", "password": "pass"},
    )

    # 后续请求自动携带 Cookie
    response = await client.get("https://example.com/dashboard")
```

### 手动管理 Cookie

```python
cookies = httpx.Cookies()
cookies.set("session_id", "abc123", domain="example.com")

async with httpx.AsyncClient(cookies=cookies) as client:
    response = await client.get("https://example.com/")
```

### 保存/加载 Cookie

```python
import json

# 保存
with open("cookies.json", "w") as f:
    json.dump(dict(client.cookies), f)

# 加载
with open("cookies.json") as f:
    cookies = json.load(f)

client = httpx.AsyncClient(cookies=cookies)
```

## 3. Session 持久化

```python
import pickle
from pathlib import Path

class SessionManager:
    def __init__(self, path: str = "session.pkl"):
        self.path = Path(path)
        self.cookies = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with self.path.open("rb") as f:
                self.cookies = pickle.load(f)

    def save(self):
        with self.path.open("wb") as f:
            pickle.dump(self.cookies, f)

    def get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(cookies=self.cookies)
```

## 4. 分布式爬虫概念

### 架构

```
┌─────────────┐
│   调度器     │  ← 分配任务
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐
│ 爬虫1│ │ 爬虫2│  ← 多个爬虫实例
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
┌──────▼──────┐
│  去重服务    │  ← Redis/BloomFilter
└──────┬──────┘
       │
┌──────▼──────┐
│  数据存储    │  ← MongoDB/PostgreSQL
└─────────────┘
```

### 关键组件

1. **URL 调度** - Redis 队列
2. **分布式去重** - Redis Set / Bloom Filter
3. **数据存储** - 数据库
4. **监控** - 进度、错误率

### Redis 队列示例

```python
import redis

class RedisQueue:
    def __init__(self, name: str, host: str = "localhost"):
        self.redis = redis.Redis(host=host)
        self.name = name

    def push(self, url: str):
        self.redis.lpush(self.name, url)

    def pop(self) -> str | None:
        result = self.redis.rpop(self.name)
        return result.decode() if result else None

    def size(self) -> int:
        return self.redis.llen(self.name)
```

### Redis 去重

```python
class RedisDedup:
    def __init__(self, name: str, host: str = "localhost"):
        self.redis = redis.Redis(host=host)
        self.name = name

    def add(self, url: str) -> bool:
        """返回 True 如果是新 URL"""
        return self.redis.sadd(self.name, url) == 1

    def contains(self, url: str) -> bool:
        return self.redis.sismember(self.name, url)
```

## 5. 增量爬取

### 基于时间戳

```python
class IncrementalCrawler:
    def __init__(self, last_crawl_time: float):
        self.last_crawl_time = last_crawl_time

    async def crawl(self, url: str):
        result = await fetcher.fetch(url)

        # 检查更新时间
        last_modified = result.headers.get("Last-Modified")
        if last_modified:
            modified_time = parse_http_date(last_modified)
            if modified_time < self.last_crawl_time:
                return None  # 跳过未更新的页面

        return result
```

### 基于 ETag

```python
class ETagCache:
    def __init__(self):
        self._etags = {}

    async def fetch_if_modified(self, url: str, client: httpx.AsyncClient):
        headers = {}
        if url in self._etags:
            headers["If-None-Match"] = self._etags[url]

        response = await client.get(url, headers=headers)

        if response.status_code == 304:
            return None  # 未修改

        if "ETag" in response.headers:
            self._etags[url] = response.headers["ETag"]

        return response
```

## 6. 反爬策略应对

### 合规方式

| 反爬手段 | 合规应对 |
|----------|----------|
| IP 限制 | 降低请求频率 |
| 验证码 | 手动处理或放弃 |
| User-Agent 检测 | 使用真实 UA |
| 请求频率检测 | 遵守 Crawl-delay |

### 技术细节

```python
# 随机请求间隔
import random

delay = random.uniform(1.0, 3.0)
await asyncio.sleep(delay)

# 请求头轮换
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",
]
ua = random.choice(user_agents)
```

## 7. 监控与告警

### 统计指标

```python
from dataclasses import dataclass
from time import time

@dataclass
class CrawlMetrics:
    start_time: float = 0.0
    pages_crawled: int = 0
    pages_failed: int = 0
    items_saved: int = 0

    @property
    def success_rate(self) -> float:
        total = self.pages_crawled + self.pages_failed
        return self.pages_crawled / total if total > 0 else 0

    @property
    def pages_per_minute(self) -> float:
        elapsed = (time() - self.start_time) / 60
        return self.pages_crawled / elapsed if elapsed > 0 else 0
```

### 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crawl.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("scraper")
logger.info(f"Crawled: {url}")
logger.error(f"Failed: {url} - {error}")
```

## 小结

| 话题 | 关键技术 |
|------|----------|
| 代理池 | ProxyRotator |
| Cookie/Session | httpx.Cookies |
| 分布式 | Redis 队列 + 去重 |
| 增量爬取 | ETag / Last-Modified |
| 监控 | 日志 + 指标 |

