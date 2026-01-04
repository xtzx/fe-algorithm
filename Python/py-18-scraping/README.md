# P18: 爬虫工程化

> 构建生产级爬虫，强调合规、可测试、可恢复

## 🎯 学完后能做

- 编写合规的爬虫
- 处理反爬和异常
- 构建可测试的爬虫

## 🚀 快速开始

```bash
# 进入项目目录
cd py-18-scraping

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行示例
python -m scraper crawl https://example.com --max-pages 10
```

## 📁 目录结构

```
py-18-scraping/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-basics.md             # 爬虫基础
│   ├── 02-compliance.md         # 合规与道德
│   ├── 03-engineering.md        # 工程化设计
│   ├── 04-testing.md            # 可测试设计
│   ├── 05-advanced.md           # 高级话题
│   ├── 06-exercises.md          # 练习题
│   ├── 07-interview.md          # 面试题
│   └── 08-playwright.md         # Playwright 动态爬取 ⭐
├── src/scraper/
│   ├── __init__.py
│   ├── fetcher.py               # 请求获取
│   ├── parser.py                # 页面解析
│   ├── pipeline.py              # 数据管道
│   ├── dedup.py                 # URL 去重
│   ├── state.py                 # 状态管理
│   ├── robots.py                # robots.txt 解析
│   └── cli.py                   # 命令行接口
├── tests/
│   ├── conftest.py
│   ├── fixtures/                # HTML 样本
│   │   ├── simple.html
│   │   └── article.html
│   ├── test_fetcher.py
│   ├── test_parser.py
│   └── test_dedup.py
├── examples/
│   └── simple_crawler.py
└── scripts/
    └── run_crawler.sh
```

## 🆚 Python vs JavaScript 对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| HTTP 请求 | `httpx` | `axios` / `fetch` |
| HTML 解析 | `BeautifulSoup` | `cheerio` |
| CSS 选择器 | `soup.select()` | `$()` |
| 动态页面 | `Playwright` | `Puppeteer` |
| 队列 | `asyncio.Queue` | `p-queue` |

## 🔧 核心功能

### 1. 基础爬取

```python
from scraper import Crawler

crawler = Crawler(
    start_url="https://example.com",
    max_pages=100,
    delay=1.0,  # 请求间隔
)

async for item in crawler.crawl():
    print(item)
```

### 2. 合规设置

```python
from scraper import Crawler, RobotsChecker

# 遵守 robots.txt
crawler = Crawler(
    start_url="https://example.com",
    respect_robots=True,
    user_agent="MyBot/1.0 (+https://example.com/bot)",
)
```

### 3. 断点续爬

```python
from scraper import Crawler, FileState

# 使用文件保存状态
crawler = Crawler(
    start_url="https://example.com",
    state=FileState("crawl_state.json"),
)

# 中断后可以恢复
await crawler.crawl()
```

### 4. 数据管道

```python
from scraper import Pipeline, JsonLineWriter

pipeline = Pipeline([
    JsonLineWriter("items.jsonl"),
])

async for item in crawler.crawl():
    await pipeline.process(item)
```

## 📚 学习路径

### 基础篇

1. **爬虫基础** - httpx + BeautifulSoup
2. **合规与道德** - robots.txt、频率限制
3. **工程化设计** - 去重、断点续爬
4. **可测试设计** - 解析函数纯函数化
5. **高级话题** - 代理池、分布式

### 动态页面

6. [Playwright 动态爬取](docs/08-playwright.md) ⭐ - 浏览器自动化、JS 渲染页面、网络拦截

## ⚠️ 重要提醒

### 合规原则

1. **遵守 robots.txt** - 检查允许爬取的路径
2. **控制请求频率** - 不要给服务器造成压力
3. **设置 User-Agent** - 表明爬虫身份
4. **尊重 Terms of Service** - 阅读网站条款
5. **只抓取公开数据** - 不要绕过登录/验证

### 法律风险

- 未经授权访问可能违法
- 大规模抓取可能构成侵权
- 爬取个人数据需遵守隐私法规
- 商业使用需特别注意

## ✅ 功能清单

- [x] 静态页面抓取
- [x] HTML 解析（CSS 选择器）
- [x] robots.txt 解析
- [x] 请求频率限制
- [x] User-Agent 设置
- [x] URL 去重
- [x] 断点续爬
- [x] 失败重试
- [x] 数据持久化（JSONL）
- [x] 可测试设计
- [x] Mock 网络请求
- [x] **Playwright 动态爬取** ⭐
- [x] **浏览器自动化**
- [x] **网络请求拦截**
- [x] **登录状态管理**

