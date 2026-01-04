# Playwright 动态页面爬取

> 现代浏览器自动化，处理 JavaScript 渲染页面

## 什么是 Playwright

Playwright 是 Microsoft 开发的浏览器自动化库，支持 Chromium、Firefox、WebKit 三大浏览器引擎。

### 为什么选择 Playwright

| 特性 | Playwright | Selenium | Puppeteer |
|------|-----------|----------|-----------|
| 多浏览器支持 | ✅ Chromium/Firefox/WebKit | ✅ 多种 | ❌ 仅 Chromium |
| 自动等待 | ✅ 内置 | ❌ 需手动 | 部分 |
| 网络拦截 | ✅ 强大 | 有限 | ✅ |
| 速度 | 快 | 较慢 | 快 |
| 无头模式 | ✅ | ✅ | ✅ |
| Python 支持 | ✅ 原生 | ✅ | ❌ Node.js |

---

## 安装

```bash
pip install playwright

# 安装浏览器（约 400MB）
playwright install

# 只安装 Chromium
playwright install chromium
```

---

## 基础用法

### 同步 API

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # 启动浏览器
    browser = p.chromium.launch(headless=True)  # headless=False 显示界面

    # 创建页面
    page = browser.new_page()

    # 访问网页
    page.goto('https://example.com')

    # 获取内容
    print(page.title())
    print(page.content())

    # 截图
    page.screenshot(path='screenshot.png')

    # 关闭
    browser.close()
```

### 异步 API（推荐）

```python
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto('https://example.com')
        print(await page.title())

        await browser.close()

asyncio.run(main())
```

---

## 页面导航

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    # 导航
    page.goto('https://example.com')
    page.goto('https://example.com', wait_until='domcontentloaded')
    page.goto('https://example.com', wait_until='networkidle')

    # 等待选项
    # 'load': 等待 load 事件（默认）
    # 'domcontentloaded': DOM 加载完成
    # 'networkidle': 网络空闲（500ms 无请求）
    # 'commit': 收到响应

    # 前进/后退
    page.go_back()
    page.go_forward()

    # 刷新
    page.reload()

    browser.close()
```

---

## 元素定位

### 定位器（Locator）

```python
# 推荐使用 Locator API（自动等待、自动重试）
page.locator('css=button')
page.locator('text=Submit')
page.locator('#id')
page.locator('.class')
page.locator('xpath=//div[@class="content"]')

# 链式定位
page.locator('div.container').locator('button')

# 内置快捷方法
page.get_by_text('Submit')
page.get_by_role('button', name='Submit')
page.get_by_placeholder('Enter your name')
page.get_by_label('Username')
page.get_by_test_id('submit-btn')  # data-testid 属性
```

### 常用选择器

```python
# CSS 选择器
page.locator('#login-btn')
page.locator('.nav-item')
page.locator('div.container > p')
page.locator('[data-id="123"]')

# 文本选择器
page.locator('text=登录')
page.locator('text="精确匹配"')
page.locator('text=/正则.*匹配/')

# XPath
page.locator('xpath=//div[@class="content"]//a')

# 组合选择器
page.locator('button:has-text("Submit")')
page.locator('div:has(img)')
page.locator('article:has-text("Python")')
```

---

## 页面交互

### 点击

```python
# 基础点击
page.locator('button').click()

# 选项
page.locator('button').click(
    button='left',      # left, right, middle
    click_count=2,      # 双击
    delay=100,          # 按住时间(ms)
    force=True,         # 强制点击（跳过可见性检查）
    position={'x': 10, 'y': 10}  # 相对位置
)

# 快捷方法
page.click('button')
page.dblclick('button')
```

### 输入

```python
# 填充（清空后输入）
page.locator('input').fill('Hello')

# 逐字输入（模拟键盘）
page.locator('input').type('Hello', delay=100)

# 清空
page.locator('input').clear()

# 按键
page.locator('input').press('Enter')
page.locator('input').press('Control+a')
```

### 选择框

```python
# 下拉选择
page.locator('select').select_option('value1')
page.locator('select').select_option(label='选项一')
page.locator('select').select_option(index=0)

# 多选
page.locator('select').select_option(['value1', 'value2'])

# 复选框
page.locator('input[type="checkbox"]').check()
page.locator('input[type="checkbox"]').uncheck()
page.locator('input[type="checkbox"]').set_checked(True)

# 单选框
page.locator('input[type="radio"][value="option1"]').check()
```

### 文件上传

```python
# 单文件
page.locator('input[type="file"]').set_input_files('path/to/file.txt')

# 多文件
page.locator('input[type="file"]').set_input_files([
    'path/to/file1.txt',
    'path/to/file2.txt'
])

# 清除选择
page.locator('input[type="file"]').set_input_files([])
```

---

## 等待策略

### 自动等待

```python
# Locator 自动等待元素可操作
page.locator('button').click()  # 自动等待按钮可点击
page.locator('input').fill('text')  # 自动等待输入框可用
```

### 显式等待

```python
# 等待选择器
page.wait_for_selector('.loaded')
page.wait_for_selector('.loaded', state='visible')
page.wait_for_selector('.loaded', state='hidden')
page.wait_for_selector('.loaded', timeout=5000)

# state 选项:
# 'attached': 存在于 DOM
# 'detached': 从 DOM 移除
# 'visible': 可见
# 'hidden': 隐藏

# 等待加载状态
page.wait_for_load_state('domcontentloaded')
page.wait_for_load_state('networkidle')

# 等待 URL
page.wait_for_url('**/login')
page.wait_for_url(lambda url: 'success' in url)

# 等待函数
page.wait_for_function('window.loaded === true')

# 固定等待（不推荐）
page.wait_for_timeout(1000)  # 1 秒
```

### 自定义等待

```python
from playwright.sync_api import expect

# 断言式等待
expect(page.locator('h1')).to_have_text('Welcome')
expect(page.locator('button')).to_be_enabled()
expect(page.locator('.loading')).to_be_hidden()
```

---

## 提取数据

### 获取文本

```python
# 单个元素
text = page.locator('h1').inner_text()
text = page.locator('h1').text_content()  # 包含隐藏文本

# 多个元素
texts = page.locator('.item').all_inner_texts()

# 获取属性
href = page.locator('a').get_attribute('href')

# 获取 HTML
html = page.locator('div').inner_html()
outer_html = page.locator('div').evaluate('el => el.outerHTML')
```

### 获取多个元素

```python
# 获取所有匹配元素
items = page.locator('.product').all()
for item in items:
    name = item.locator('.name').inner_text()
    price = item.locator('.price').inner_text()
    print(f"{name}: {price}")

# 计数
count = page.locator('.item').count()

# 第 N 个元素
page.locator('.item').nth(0)  # 第一个
page.locator('.item').first   # 第一个
page.locator('.item').last    # 最后一个
```

### 执行 JavaScript

```python
# 在页面中执行
result = page.evaluate('document.title')
result = page.evaluate('() => window.innerWidth')

# 传参
result = page.evaluate('(x) => x * 2', 5)

# 在元素上执行
text = page.locator('h1').evaluate('el => el.textContent')
```

---

## 网络拦截

### 监听请求/响应

```python
def log_request(request):
    print(f">> {request.method} {request.url}")

def log_response(response):
    print(f"<< {response.status} {response.url}")

page.on('request', log_request)
page.on('response', log_response)

page.goto('https://example.com')
```

### 拦截和修改请求

```python
def handle_route(route):
    # 阻止请求
    if 'ads' in route.request.url:
        route.abort()
        return

    # 修改请求
    headers = route.request.headers
    headers['X-Custom-Header'] = 'value'
    route.continue_(headers=headers)

# 拦截匹配的 URL
page.route('**/*.png', lambda route: route.abort())  # 阻止图片
page.route('**/api/**', handle_route)

page.goto('https://example.com')
```

### 模拟响应

```python
def mock_api(route):
    route.fulfill(
        status=200,
        content_type='application/json',
        body='{"data": "mocked"}'
    )

page.route('**/api/data', mock_api)
```

---

## 处理弹窗

```python
# 处理 alert/confirm/prompt
def handle_dialog(dialog):
    print(f"Dialog: {dialog.message}")
    dialog.accept()  # 或 dialog.dismiss()

page.on('dialog', handle_dialog)

# 处理新窗口/标签
with page.expect_popup() as popup_info:
    page.locator('a[target="_blank"]').click()
popup = popup_info.value
print(popup.url)

# 处理下载
with page.expect_download() as download_info:
    page.locator('a.download').click()
download = download_info.value
download.save_as('downloaded_file.pdf')
```

---

## 实战：爬取动态页面

### 示例：爬取 SPA 页面

```python
import asyncio
from playwright.async_api import async_playwright

async def scrape_spa():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()

        # 访问页面
        await page.goto('https://spa-example.com')

        # 等待数据加载
        await page.wait_for_selector('.product-list')

        # 滚动加载更多
        for _ in range(5):
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await page.wait_for_timeout(1000)

        # 提取数据
        products = await page.locator('.product-item').all()
        results = []

        for product in products:
            name = await product.locator('.name').inner_text()
            price = await product.locator('.price').inner_text()
            results.append({'name': name, 'price': price})

        await browser.close()
        return results

# 运行
data = asyncio.run(scrape_spa())
print(data)
```

### 示例：处理登录

```python
async def login_and_scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # 调试时显示浏览器
        page = await browser.new_page()

        # 登录
        await page.goto('https://example.com/login')
        await page.locator('#username').fill('myuser')
        await page.locator('#password').fill('mypass')
        await page.locator('button[type="submit"]').click()

        # 等待登录成功
        await page.wait_for_url('**/dashboard')

        # 保存登录状态
        await page.context.storage_state(path='auth.json')

        # 爬取需要登录的页面
        await page.goto('https://example.com/protected')
        content = await page.content()

        await browser.close()
        return content

# 复用登录状态
async def scrape_with_auth():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(storage_state='auth.json')
        page = await context.new_page()

        await page.goto('https://example.com/protected')
        # 已登录状态

        await browser.close()
```

### 示例：处理无限滚动

```python
async def scrape_infinite_scroll():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto('https://example.com/feed')

        all_items = []
        prev_height = 0

        while True:
            # 滚动到底部
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await page.wait_for_timeout(2000)

            # 获取当前所有项目
            items = await page.locator('.feed-item').all_inner_texts()

            # 检查是否有新内容
            curr_height = await page.evaluate('document.body.scrollHeight')
            if curr_height == prev_height:
                break
            prev_height = curr_height

            # 限制数量
            if len(items) >= 100:
                break

        all_items = await page.locator('.feed-item').all_inner_texts()
        await browser.close()
        return all_items
```

---

## 并发爬取

```python
import asyncio
from playwright.async_api import async_playwright

async def scrape_page(browser, url):
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=30000)
        title = await page.title()
        return {'url': url, 'title': title}
    except Exception as e:
        return {'url': url, 'error': str(e)}
    finally:
        await page.close()

async def main():
    urls = [
        'https://example1.com',
        'https://example2.com',
        'https://example3.com',
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # 并发限制
        semaphore = asyncio.Semaphore(5)

        async def limited_scrape(url):
            async with semaphore:
                return await scrape_page(browser, url)

        results = await asyncio.gather(*[limited_scrape(url) for url in urls])

        await browser.close()
        return results

results = asyncio.run(main())
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 元素未加载 | 页面动态加载 | 使用 `wait_for_selector` |
| 反爬检测 | 被识别为机器人 | 设置 User-Agent、使用代理 |
| 内存泄漏 | 未关闭页面/浏览器 | 使用 `with` 语句或确保关闭 |
| 超时 | 网络慢或元素不存在 | 设置合理的 `timeout` |
| headless 差异 | 有头无头行为不同 | 调试时用 `headless=False` |

---

## 反检测技巧

```python
async with async_playwright() as p:
    browser = await p.chromium.launch(
        headless=True,
        args=[
            '--disable-blink-features=AutomationControlled',
        ]
    )

    context = await browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...',
        viewport={'width': 1920, 'height': 1080},
        locale='zh-CN',
        timezone_id='Asia/Shanghai',
    )

    page = await context.new_page()

    # 隐藏 webdriver 属性
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)
```

---

## 小结

1. **Playwright 优势**：自动等待、多浏览器、异步支持
2. **Locator API**：推荐使用，自动等待和重试
3. **网络拦截**：监听、阻止、模拟请求
4. **登录状态**：`storage_state` 保存和复用
5. **并发控制**：使用 `Semaphore` 限制
6. **反检测**：User-Agent、隐藏 webdriver 属性

