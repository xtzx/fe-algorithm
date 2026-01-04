# 合规与道德

> robots.txt、请求频率、User-Agent、法律边界

## 1. robots.txt

### 什么是 robots.txt

```
https://example.com/robots.txt

User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /public/

Crawl-delay: 10

Sitemap: https://example.com/sitemap.xml
```

### 规则解读

| 指令 | 含义 |
|------|------|
| User-agent | 适用的爬虫 |
| Disallow | 禁止访问的路径 |
| Allow | 允许访问的路径 |
| Crawl-delay | 请求间隔（秒） |
| Sitemap | 站点地图 URL |

### 使用 scraper 检查

```python
from scraper import RobotsChecker

checker = RobotsChecker(user_agent="MyBot/1.0")

# 检查 URL 是否允许
if await checker.is_allowed("https://example.com/page"):
    # 可以爬取
    pass

# 获取建议的请求间隔
delay = await checker.get_crawl_delay("https://example.com")
```

## 2. 请求频率限制

### 为什么限制频率

- 避免服务器过载
- 遵守网站政策
- 避免被封禁
- 职业道德

### 实现限流

```python
from scraper import RateLimitedFetcher

# 每秒最多 2 个请求
fetcher = RateLimitedFetcher(
    requests_per_second=2.0,
    jitter=0.5,  # 添加随机延迟
)
```

### 遵守 Crawl-delay

```python
from scraper import RobotsChecker

checker = RobotsChecker(user_agent="MyBot")
delay = await checker.get_crawl_delay(url)

if delay:
    await asyncio.sleep(delay)
```

## 3. User-Agent 设置

### 为什么设置 User-Agent

- 表明爬虫身份
- 便于网站管理员联系
- 遵守 robots.txt 规则

### 最佳实践

```python
# ✅ 推荐：包含机器人名称和联系方式
user_agent = "MyBot/1.0 (+https://example.com/bot)"

# ❌ 不推荐：伪装成浏览器
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
```

### 在 scraper 中使用

```python
from scraper import Fetcher

fetcher = Fetcher(
    user_agent="MyBot/1.0 (+https://example.com/bot)",
)
```

## 4. 法律与道德边界

### ⚠️ 可能违法的行为

1. **绕过访问控制**
   - 破解登录
   - 绕过验证码
   - 利用漏洞

2. **侵犯版权**
   - 大量复制受保护内容
   - 商业使用爬取的内容

3. **违反服务条款**
   - 很多网站明确禁止爬取

4. **侵犯隐私**
   - 收集个人信息
   - 违反 GDPR 等法规

### ✅ 安全的做法

1. **只爬公开数据**
2. **遵守 robots.txt**
3. **控制请求频率**
4. **尊重版权**
5. **不存储个人信息**
6. **阅读服务条款**

### 灰色地带

| 场景 | 风险级别 |
|------|----------|
| 爬取公开新闻标题 | 低 |
| 爬取价格比较 | 中 |
| 大规模爬取用户数据 | 高 |
| 绕过反爬机制 | 高 |

## 5. 反爬机制

### 常见反爬手段

1. **IP 限制** - 同一 IP 请求过多被封
2. **验证码** - 验证是否为人类
3. **登录墙** - 需要登录才能访问
4. **User-Agent 检测** - 检测爬虫标识
5. **请求频率检测** - 过快则封禁

### 合规应对方式

```python
# 1. 降低请求频率
fetcher = RateLimitedFetcher(requests_per_second=0.5)

# 2. 设置合理的 User-Agent
fetcher = Fetcher(user_agent="MyBot/1.0 (+contact@example.com)")

# 3. 遵守 robots.txt
if await robots_checker.is_allowed(url):
    await fetcher.fetch(url)
```

### ❌ 不推荐的做法

```python
# 不推荐：使用代理绕过 IP 限制
# 不推荐：伪装 User-Agent
# 不推荐：自动识别验证码
```

## 6. 最佳实践清单

### 开始爬取前

- [ ] 阅读网站的 robots.txt
- [ ] 检查网站的服务条款
- [ ] 确认数据的使用目的合法
- [ ] 设置合理的请求间隔

### 爬取过程中

- [ ] 使用标识性 User-Agent
- [ ] 遵守 Crawl-delay
- [ ] 处理错误而不是重试轰炸
- [ ] 记录爬取日志

### 爬取后

- [ ] 妥善存储数据
- [ ] 不公开敏感信息
- [ ] 遵守数据使用约定

## 小结

| 原则 | 做法 |
|------|------|
| 遵守 robots.txt | 检查并遵守规则 |
| 控制频率 | 1-2 请求/秒 |
| 表明身份 | 设置 User-Agent |
| 尊重版权 | 不大量复制内容 |
| 遵守法律 | 阅读服务条款 |

