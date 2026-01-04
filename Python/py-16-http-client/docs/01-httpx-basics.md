# httpx 基础

> 掌握现代 Python HTTP 客户端

## 1. 为什么选择 httpx

| 特性 | httpx | requests |
|------|-------|----------|
| 异步支持 | ✓ 原生 async/await | ✗ |
| HTTP/2 | ✓ | ✗ |
| 类型注解 | ✓ 完整 | 部分 |
| 维护状态 | 活跃 | 维护模式 |
| API 兼容 | 兼容 requests | - |

## 2. 安装

```bash
pip install httpx
```

## 3. 基础请求

### 同步请求

```python
import httpx

# 简单 GET
response = httpx.get("https://api.example.com/users")
print(response.status_code)  # 200
print(response.json())       # [{"id": 1, "name": "Alice"}]

# 带参数的 GET
response = httpx.get(
    "https://api.example.com/users",
    params={"page": 1, "limit": 10},
)

# POST JSON
response = httpx.post(
    "https://api.example.com/users",
    json={"name": "Bob", "email": "bob@example.com"},
)

# POST 表单
response = httpx.post(
    "https://api.example.com/login",
    data={"username": "alice", "password": "secret"},
)

# 其他方法
httpx.put("https://api.example.com/users/1", json={"name": "Updated"})
httpx.patch("https://api.example.com/users/1", json={"name": "Patched"})
httpx.delete("https://api.example.com/users/1")
```

### 异步请求

```python
import asyncio
import httpx

async def fetch_users():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/users")
        return response.json()

# 运行
users = asyncio.run(fetch_users())
```

## 4. 使用客户端实例

推荐使用客户端实例（连接复用）：

```python
# 同步
with httpx.Client() as client:
    response = client.get("https://api.example.com/users")

# 异步
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com/users")
```

### 配置 base_url

```python
client = httpx.Client(base_url="https://api.example.com")
response = client.get("/users")  # 完整 URL: https://api.example.com/users
```

### 默认请求头

```python
client = httpx.Client(
    base_url="https://api.example.com",
    headers={
        "Authorization": "Bearer token123",
        "Accept": "application/json",
    },
)
```

## 5. 请求参数

### 查询参数

```python
# 字典
response = client.get("/search", params={"q": "python", "page": 1})

# 列表（多值）
response = client.get("/filter", params={"tag": ["python", "web"]})
# URL: /filter?tag=python&tag=web
```

### 请求头

```python
response = client.get(
    "/users",
    headers={
        "X-Custom-Header": "value",
        "Accept-Language": "zh-CN",
    },
)
```

### JSON Body

```python
response = client.post(
    "/users",
    json={"name": "Alice", "age": 30},
)
```

### 表单数据

```python
response = client.post(
    "/login",
    data={"username": "alice", "password": "secret"},
)
```

### 文件上传

```python
# 单文件
with open("file.txt", "rb") as f:
    response = client.post("/upload", files={"file": f})

# 多文件
response = client.post(
    "/upload",
    files=[
        ("files", ("file1.txt", open("file1.txt", "rb"))),
        ("files", ("file2.txt", open("file2.txt", "rb"))),
    ],
)
```

## 6. 响应处理

```python
response = client.get("/users")

# 状态码
response.status_code      # 200
response.is_success       # True (2xx)
response.is_error         # False

# 响应头
response.headers          # Headers object
response.headers["content-type"]  # "application/json"

# 响应体
response.text             # 文本
response.json()           # JSON → dict/list
response.content          # bytes
response.read()           # 异步读取全部

# 编码
response.encoding         # "utf-8"

# URL
response.url              # 最终 URL（跟随重定向后）
```

### 状态码检查

```python
response = client.get("/users")

# 自动抛出异常
response.raise_for_status()  # 4xx/5xx 会抛出 HTTPStatusError

# 手动检查
if response.is_success:
    data = response.json()
elif response.status_code == 404:
    print("Not found")
else:
    print(f"Error: {response.status_code}")
```

## 7. 与 JS fetch 对比

```python
# Python httpx
import httpx

response = httpx.post(
    "https://api.example.com/users",
    json={"name": "Alice"},
    headers={"Authorization": "Bearer token"},
)
data = response.json()
```

```javascript
// JavaScript fetch
const response = await fetch("https://api.example.com/users", {
  method: "POST",
  headers: {
    "Authorization": "Bearer token",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ name: "Alice" }),
});
const data = await response.json();
```

## 8. 常见错误处理

```python
import httpx

try:
    response = client.get("/users")
    response.raise_for_status()
    data = response.json()
except httpx.ConnectError:
    print("无法连接到服务器")
except httpx.TimeoutException:
    print("请求超时")
except httpx.HTTPStatusError as e:
    print(f"HTTP 错误: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"请求错误: {e}")
```

## 小结

| 概念 | 要点 |
|------|------|
| 客户端 | 推荐使用 `Client()` 复用连接 |
| 异步 | `AsyncClient` + `async/await` |
| base_url | 简化 API 调用 |
| json | 自动序列化 Python 对象 |
| raise_for_status | 4xx/5xx 抛出异常 |

