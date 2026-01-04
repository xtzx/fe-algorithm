# 高级配置

> 超时、连接池、代理、SSL/TLS

## 1. 超时配置

### 简单超时

```python
import httpx

# 所有操作共用超时
client = httpx.Client(timeout=10.0)  # 10 秒

# 单次请求覆盖
response = client.get("/slow", timeout=30.0)
```

### 细粒度超时

```python
# 分别设置不同阶段的超时
timeout = httpx.Timeout(
    connect=5.0,    # 连接超时
    read=10.0,      # 读取超时
    write=10.0,     # 写入超时
    pool=5.0,       # 等待连接池超时
)

client = httpx.Client(timeout=timeout)
```

### 禁用超时

```python
# 不推荐，但某些场景需要
client = httpx.Client(timeout=None)
```

## 2. 连接池

### 连接池配置

```python
# 自定义连接池
limits = httpx.Limits(
    max_keepalive_connections=20,  # 保持的连接数
    max_connections=100,            # 最大连接数
    keepalive_expiry=30.0,          # 连接过期时间（秒）
)

client = httpx.Client(limits=limits)
```

### 为什么需要连接池

```
无连接池:
  请求1: [DNS] [TCP连接] [TLS握手] [请求] [响应] [关闭]
  请求2: [DNS] [TCP连接] [TLS握手] [请求] [响应] [关闭]

有连接池:
  请求1: [DNS] [TCP连接] [TLS握手] [请求] [响应]
  请求2:                             [请求] [响应]  ← 复用连接
```

## 3. 代理设置

### HTTP 代理

```python
# 所有请求使用代理
client = httpx.Client(proxy="http://localhost:8080")

# 区分 HTTP/HTTPS
client = httpx.Client(
    proxies={
        "http://": "http://localhost:8080",
        "https://": "http://localhost:8081",
    }
)
```

### SOCKS 代理

```bash
pip install httpx[socks]
```

```python
client = httpx.Client(proxy="socks5://localhost:1080")
```

### 环境变量

```python
# 自动读取 HTTP_PROXY、HTTPS_PROXY 环境变量
client = httpx.Client(trust_env=True)
```

## 4. SSL/TLS 配置

### 验证证书

```python
# 默认验证
client = httpx.Client()

# 禁用验证（不推荐，仅开发环境）
client = httpx.Client(verify=False)
```

### 自定义 CA 证书

```python
# 使用自定义 CA 证书
client = httpx.Client(verify="/path/to/ca-bundle.crt")
```

### 客户端证书

```python
# mTLS（双向 TLS）
client = httpx.Client(
    cert=("/path/to/client.crt", "/path/to/client.key"),
)

# 带密码的私钥
client = httpx.Client(
    cert=("/path/to/client.crt", "/path/to/client.key", "password"),
)
```

## 5. HTTP/2

```bash
pip install httpx[http2]
```

```python
# 启用 HTTP/2
client = httpx.Client(http2=True)

# 检查协议版本
response = client.get("https://api.example.com")
print(response.http_version)  # "HTTP/2"
```

### HTTP/2 优势

- 多路复用：一个连接多个请求
- 头部压缩：减少带宽
- 服务器推送

## 6. 重定向

```python
# 默认跟随重定向
client = httpx.Client(follow_redirects=True)

# 禁用重定向
client = httpx.Client(follow_redirects=False)

# 限制重定向次数
client = httpx.Client(max_redirects=5)

# 检查重定向历史
response = client.get("/redirect")
print(response.history)  # [Response1, Response2, ...]
print(response.url)      # 最终 URL
```

## 7. Cookies

```python
# 手动设置
response = client.get("/", cookies={"session": "abc123"})

# 使用 cookies jar
client = httpx.Client()
client.cookies.set("session", "abc123")
response = client.get("/")

# 跨请求自动管理 cookies
with httpx.Client() as client:
    # 登录后 cookies 自动保存
    client.post("/login", data={"user": "alice", "pass": "secret"})
    # 后续请求自动携带 cookies
    response = client.get("/dashboard")
```

## 8. 事件钩子

```python
def log_request(request):
    print(f"Request: {request.method} {request.url}")

def log_response(response):
    print(f"Response: {response.status_code}")

client = httpx.Client(
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    }
)
```

## 9. 完整配置示例

```python
import httpx

# 生产级配置
client = httpx.Client(
    base_url="https://api.example.com",

    # 超时
    timeout=httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0,
    ),

    # 连接池
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
    ),

    # 请求头
    headers={
        "User-Agent": "MyApp/1.0",
        "Accept": "application/json",
    },

    # HTTP/2
    http2=True,

    # 重定向
    follow_redirects=True,
    max_redirects=10,
)
```

## 小结

| 配置 | 用途 | 默认值 |
|------|------|--------|
| timeout | 防止请求无限等待 | 5s |
| limits | 控制连接数量 | 100 |
| proxy | 代理服务器 | None |
| verify | SSL 验证 | True |
| http2 | HTTP/2 支持 | False |
| follow_redirects | 自动跟随重定向 | True |

