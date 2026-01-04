# P19: 综合项目 - 技术博客聚合器

> 综合运用网络和并发知识，完成数据采集项目

## 🎯 项目目标

开发一个「技术博客聚合器」：
- 从多个技术博客抓取文章列表
- 异步并发提高效率
- 数据清洗与结构化
- 生成聚合报告
- 增量更新（只抓新文章）

## 🚀 快速开始

```bash
# 进入项目目录
cd py-19-project-scraping

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行采集
python -m blog_aggregator collect --all

# 生成报告
python -m blog_aggregator report
```

## 📁 目录结构

```
py-19-project-scraping/
├── README.md
├── pyproject.toml
├── config.toml                    # 配置文件
├── src/blog_aggregator/
│   ├── __init__.py
│   ├── models.py                  # 数据模型
│   ├── sources/                   # 博客源解析器
│   │   ├── __init__.py
│   │   ├── base.py               # 基类
│   │   ├── dev_to.py             # DEV.to
│   │   ├── hashnode.py           # Hashnode
│   │   └── medium.py             # Medium（示例）
│   ├── fetcher.py                 # 并发获取器
│   ├── pipeline.py                # 数据管道
│   ├── storage.py                 # 数据存储
│   ├── reporter.py                # 报告生成
│   └── cli.py                     # 命令行接口
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   └── test_*.py
├── data/                          # 数据目录
│   ├── articles.jsonl            # 文章存储
│   └── state.json                # 状态文件
└── scripts/
    └── run_demo.sh
```

## 🔧 核心功能

### 1. 多源抓取

```python
from blog_aggregator import BlogAggregator

aggregator = BlogAggregator()

# 抓取所有配置的源
articles = await aggregator.collect_all()

# 抓取特定源
articles = await aggregator.collect(sources=["dev_to", "hashnode"])
```

### 2. 并发控制

```python
from blog_aggregator import Fetcher

fetcher = Fetcher(
    max_concurrent=10,           # 全局最大并发
    per_host_limit=3,            # 每个站点最大并发
    rate_limit=2.0,              # 每秒请求数
)
```

### 3. 数据模型

```python
from blog_aggregator.models import Article

article = Article(
    id="unique-id",
    title="Python Async Programming",
    url="https://dev.to/...",
    source="dev_to",
    author="John Doe",
    published_at=datetime.now(),
    tags=["python", "async"],
)
```

### 4. 增量更新

```python
# 只抓取新文章
articles = await aggregator.collect_all(incremental=True)
```

### 5. 报告生成

```bash
# 生成 Markdown 报告
python -m blog_aggregator report --format markdown --output report.md

# 生成 JSON 报告
python -m blog_aggregator report --format json
```

## 📊 技术架构

```
┌─────────────────┐
│     CLI         │  命令行入口
└────────┬────────┘
         │
┌────────▼────────┐
│   Aggregator    │  协调器
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Sources│ │Fetcher│  多源解析 + 并发获取
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
┌────────▼────────┐
│    Pipeline     │  数据清洗和验证
└────────┬────────┘
         │
┌────────▼────────┐
│    Storage      │  持久化 (JSONL)
└─────────────────┘
```

## ⚙️ 配置

### config.toml

```toml
[general]
data_dir = "data"
max_concurrent = 10
rate_limit = 2.0

[sources.dev_to]
enabled = true
base_url = "https://dev.to"
per_page = 30

[sources.hashnode]
enabled = true
base_url = "https://hashnode.com"
```

## 📝 命令行使用

```bash
# 采集所有启用的源
python -m blog_aggregator collect --all

# 采集特定源
python -m blog_aggregator collect --source dev_to

# 增量采集（只抓新文章）
python -m blog_aggregator collect --all --incremental

# 查看状态
python -m blog_aggregator status

# 生成报告
python -m blog_aggregator report --format markdown
```

## 🧪 知识应用

| 知识点 | 应用 |
|--------|------|
| P16 HTTP 客户端 | httpx 异步请求、重试、超时 |
| P17 asyncio | 并发控制、TaskGroup、Semaphore |
| P18 爬虫工程 | 解析、去重、断点续爬、robots.txt |
| P12 数据模型 | pydantic 验证、数据清洗 |
| P13 文件自动化 | 增量处理、状态管理 |

## ✅ 功能清单

- [x] 多源抓取（3+ 博客源）
- [x] 统一数据模型
- [x] asyncio 并发
- [x] 每站点并发限制
- [x] 全局速率限制
- [x] pydantic 模型验证
- [x] 数据清洗与规范化
- [x] URL 去重
- [x] 增量更新
- [x] JSONL 存储
- [x] 状态管理
- [x] 报告生成（Markdown/JSON）

