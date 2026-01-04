# 10. 其他常用模块

## 本节目标

- 了解常用工具模块
- 掌握 random、uuid、hashlib 等

---

## random - 随机数

```python
import random

# 随机浮点数 [0, 1)
print(random.random())

# 指定范围
print(random.uniform(1.0, 10.0))    # 浮点数
print(random.randint(1, 100))       # 整数（包含端点）
print(random.randrange(0, 100, 5))  # 步长

# 序列操作
items = [1, 2, 3, 4, 5]
print(random.choice(items))         # 随机选一个
print(random.choices(items, k=3))   # 可重复选 k 个
print(random.sample(items, k=3))    # 不重复选 k 个

# 打乱顺序
random.shuffle(items)
print(items)

# 设置种子（可复现）
random.seed(42)
```

---

## secrets - 安全随机

**用于密码、令牌等安全场景**，比 random 更安全。

```python
import secrets

# 随机字节
token = secrets.token_bytes(16)
print(token)

# 十六进制字符串
token = secrets.token_hex(16)
print(token)  # 32 个字符

# URL 安全字符串
token = secrets.token_urlsafe(16)
print(token)

# 随机整数
n = secrets.randbelow(100)  # [0, 100)

# 安全选择
items = ["apple", "banana", "cherry"]
choice = secrets.choice(items)

# 比较字符串（防止时序攻击）
secrets.compare_digest("password123", user_input)
```

### random vs secrets

| 场景 | 使用 |
|------|------|
| 游戏、模拟 | `random` |
| 密码、令牌 | `secrets` |
| 加密密钥 | `secrets` |
| 验证码 | `secrets` |

---

## uuid - 唯一标识

```python
import uuid

# UUID4（随机生成，最常用）
id1 = uuid.uuid4()
print(id1)  # 如 a3b8f042-1e16-4f0c-8a5c-7c6df4d3b9e5

# UUID1（基于时间和 MAC 地址）
id2 = uuid.uuid1()

# 从字符串创建
id3 = uuid.UUID("a3b8f042-1e16-4f0c-8a5c-7c6df4d3b9e5")

# 属性
print(id1.hex)        # 不带连字符
print(id1.bytes)      # 16 字节
print(str(id1))       # 标准格式
```

---

## hashlib - 哈希

```python
import hashlib

# MD5（不安全，仅用于校验）
md5 = hashlib.md5(b"hello")
print(md5.hexdigest())  # 5d41402abc4b2a76b9719d911017c592

# SHA256（推荐）
sha256 = hashlib.sha256(b"hello")
print(sha256.hexdigest())

# SHA512
sha512 = hashlib.sha512(b"hello")
print(sha512.hexdigest())

# 增量更新
h = hashlib.sha256()
h.update(b"hello ")
h.update(b"world")
print(h.hexdigest())

# 大文件哈希
def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

### 密码哈希

```python
import hashlib

# 带盐值（推荐使用 bcrypt 或 argon2）
def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

# 更好的方式：使用 pbkdf2
import secrets

def hash_password_secure(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000  # 迭代次数
    )
    return salt, key.hex()
```

---

## base64 - 编解码

```python
import base64

# 编码
data = b"Hello, World!"
encoded = base64.b64encode(data)
print(encoded)  # b'SGVsbG8sIFdvcmxkIQ=='

# 解码
decoded = base64.b64decode(encoded)
print(decoded)  # b'Hello, World!'

# URL 安全编码
url_safe = base64.urlsafe_b64encode(data)
print(url_safe)

# 字符串版本
text = "Hello"
encoded_str = base64.b64encode(text.encode()).decode()
print(encoded_str)  # SGVsbG8=
```

---

## urllib - URL 处理

```python
from urllib.parse import urlparse, urlencode, parse_qs, urljoin, quote, unquote

# 解析 URL
url = "https://example.com:8080/path?name=alice&age=25#section"
result = urlparse(url)
print(result.scheme)    # https
print(result.netloc)    # example.com:8080
print(result.path)      # /path
print(result.query)     # name=alice&age=25
print(result.fragment)  # section

# 解析查询参数
params = parse_qs(result.query)
print(params)  # {'name': ['alice'], 'age': ['25']}

# 构建查询字符串
query = urlencode({"name": "张三", "age": 25})
print(query)  # name=%E5%BC%A0%E4%B8%89&age=25

# URL 拼接
base = "https://example.com/api/"
full = urljoin(base, "users/123")
print(full)  # https://example.com/api/users/123

# URL 编解码
encoded = quote("你好 世界")
print(encoded)  # %E4%BD%A0%E5%A5%BD%20%E4%B8%96%E7%95%8C

decoded = unquote(encoded)
print(decoded)  # 你好 世界
```

### urllib.request - 发送请求

```python
from urllib.request import urlopen, Request
import json

# 简单 GET
with urlopen("https://api.github.com") as response:
    data = response.read()
    print(json.loads(data))

# 带 headers
req = Request(
    "https://api.github.com",
    headers={"User-Agent": "Python"}
)
with urlopen(req) as response:
    data = response.read()
```

**注意**：实际项目推荐使用 `requests` 或 `httpx` 库。

---

## struct - 二进制数据

```python
import struct

# 打包
data = struct.pack("ihf", 42, 1000, 3.14)
print(data)

# 解包
values = struct.unpack("ihf", data)
print(values)  # (42, 1000, 3.140000104904175)

# 格式符：
# i - int (4 bytes)
# h - short (2 bytes)
# f - float (4 bytes)
# d - double (8 bytes)
# s - string
```

---

## itertools（回顾）

```python
from itertools import (
    count, cycle, repeat,           # 无限迭代
    chain, zip_longest,             # 组合
    groupby, combinations, permutations,  # 分组排列
    takewhile, dropwhile            # 筛选
)

# 常用示例
print(list(chain([1, 2], [3, 4])))  # [1, 2, 3, 4]
print(list(combinations("ABC", 2)))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]
```

---

## functools（回顾）

```python
from functools import partial, lru_cache, reduce

# partial - 偏函数
from operator import mul
double = partial(mul, 2)
print(double(5))  # 10

# lru_cache - 缓存
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# reduce
from functools import reduce
result = reduce(lambda x, y: x + y, [1, 2, 3, 4])
print(result)  # 10
```

---

## 本节要点

1. `random` 普通随机，`secrets` 安全随机
2. `uuid.uuid4()` 生成唯一 ID
3. `hashlib.sha256()` 计算哈希
4. `base64` 编解码
5. `urllib.parse` 解析和构建 URL


