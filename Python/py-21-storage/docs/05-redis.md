# Redis 缓存

## 概述

Redis 是高性能的键值存储，常用于：

1. **缓存** - 减少数据库压力
2. **会话存储** - 用户登录状态
3. **分布式锁** - 并发控制
4. **限流** - API 访问控制
5. **消息队列** - 异步任务

## 1. 安装和连接

```bash
pip install redis
```

```python
import redis

# 连接
client = redis.Redis(host='localhost', port=6379, db=0)

# 或使用 URL
client = redis.from_url("redis://localhost:6379/0")

# 连接池
pool = redis.ConnectionPool.from_url("redis://localhost:6379/0")
client = redis.Redis(connection_pool=pool)

# 检查连接
client.ping()  # True
```

## 2. 数据类型

### 2.1 String（字符串）

```python
# 设置
client.set("name", "John")
client.setex("session", 3600, "data")  # 带过期时间

# 获取
client.get("name")  # b"John"

# 批量操作
client.mset({"key1": "v1", "key2": "v2"})
client.mget("key1", "key2")  # [b"v1", b"v2"]

# 计数
client.incr("counter")  # 1
client.incrby("counter", 5)  # 6
client.decr("counter")  # 5
```

### 2.2 Hash（哈希）

```python
# 设置
client.hset("user:1", "name", "John")
client.hset("user:1", mapping={"age": 30, "city": "NYC"})

# 获取
client.hget("user:1", "name")  # b"John"
client.hgetall("user:1")  # {b"name": b"John", b"age": b"30"}

# 删除
client.hdel("user:1", "city")

# 检查
client.hexists("user:1", "name")  # True
```

### 2.3 List（列表）

```python
# 插入
client.lpush("queue", "task1", "task2")  # 左侧
client.rpush("queue", "task3")  # 右侧

# 弹出
client.lpop("queue")  # b"task2"
client.rpop("queue")  # b"task3"
client.blpop("queue", timeout=5)  # 阻塞式

# 范围
client.lrange("queue", 0, -1)  # 获取所有

# 长度
client.llen("queue")
```

### 2.4 Set（集合）

```python
# 添加
client.sadd("tags", "python", "redis", "database")

# 获取
client.smembers("tags")  # {b"python", b"redis", b"database"}

# 检查
client.sismember("tags", "python")  # True

# 运算
client.sinter("tags1", "tags2")  # 交集
client.sunion("tags1", "tags2")  # 并集
client.sdiff("tags1", "tags2")  # 差集
```

### 2.5 Sorted Set（有序集合）

```python
# 添加（分数用于排序）
client.zadd("leaderboard", {"player1": 100, "player2": 200})

# 获取（按分数）
client.zrange("leaderboard", 0, -1)  # 升序
client.zrevrange("leaderboard", 0, -1)  # 降序
client.zrange("leaderboard", 0, -1, withscores=True)  # 带分数

# 排名
client.zrank("leaderboard", "player1")  # 排名（从0开始）

# 分数范围
client.zrangebyscore("leaderboard", 100, 200)
```

## 3. 缓存策略

### 3.1 Cache-Aside（旁路缓存）

```python
def get_user(user_id: int):
    # 1. 先查缓存
    cache_key = f"user:{user_id}"
    cached = client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. 缓存未命中，查数据库
    user = db.get_user(user_id)
    if user:
        # 3. 写入缓存
        client.setex(cache_key, 300, json.dumps(user))

    return user

def update_user(user_id: int, data: dict):
    # 1. 更新数据库
    db.update_user(user_id, data)

    # 2. 删除缓存（或更新）
    cache_key = f"user:{user_id}"
    client.delete(cache_key)
```

### 3.2 Write-Through（写穿透）

```python
def update_user(user_id: int, data: dict):
    # 1. 更新数据库
    user = db.update_user(user_id, data)

    # 2. 更新缓存
    cache_key = f"user:{user_id}"
    client.setex(cache_key, 300, json.dumps(user))

    return user
```

### 3.3 缓存装饰器

```python
import functools
import json

def cached(ttl: int = 300, prefix: str = "cache"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = f"{prefix}:{func.__name__}:{args}:{kwargs}"

            # 查缓存
            cached = client.get(key)
            if cached:
                return json.loads(cached)

            # 执行函数
            result = func(*args, **kwargs)

            # 存缓存
            if result is not None:
                client.setex(key, ttl, json.dumps(result))

            return result
        return wrapper
    return decorator

# 使用
@cached(ttl=60)
def get_user(user_id: int):
    return db.get_user(user_id)
```

## 4. 缓存问题

### 4.1 缓存穿透

**问题**：查询不存在的数据，每次都穿透到数据库

**解决方案**：

```python
def get_user(user_id: int):
    cache_key = f"user:{user_id}"
    cached = client.get(cache_key)

    if cached == b"NULL":
        return None  # 空值标记
    if cached:
        return json.loads(cached)

    user = db.get_user(user_id)
    if user:
        client.setex(cache_key, 300, json.dumps(user))
    else:
        # 缓存空值，短过期时间
        client.setex(cache_key, 60, "NULL")

    return user
```

### 4.2 缓存雪崩

**问题**：大量缓存同时过期，导致数据库压力激增

**解决方案**：

```python
import random

def cache_with_jitter(key: str, value: str, base_ttl: int):
    # 添加随机抖动
    jitter = random.randint(0, 60)
    ttl = base_ttl + jitter
    client.setex(key, ttl, value)
```

### 4.3 缓存击穿

**问题**：热点数据过期瞬间，大量请求同时查询数据库

**解决方案**：使用分布式锁

```python
def get_hot_data(key: str):
    cached = client.get(key)
    if cached:
        return json.loads(cached)

    # 获取锁
    lock_key = f"lock:{key}"
    if client.set(lock_key, "1", nx=True, ex=10):
        try:
            # 双重检查
            cached = client.get(key)
            if cached:
                return json.loads(cached)

            # 查询数据库
            data = db.get_data(key)
            client.setex(key, 300, json.dumps(data))
            return data
        finally:
            client.delete(lock_key)
    else:
        # 等待并重试
        time.sleep(0.1)
        return get_hot_data(key)
```

## 5. 分布式锁

```python
import uuid

class DistributedLock:
    def __init__(self, client, name, timeout=10):
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = timeout
        self.token = str(uuid.uuid4())

    def acquire(self):
        return self.client.set(
            self.name,
            self.token,
            nx=True,
            ex=self.timeout,
        )

    def release(self):
        # Lua 脚本保证原子性
        lua = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        return self.client.eval(lua, 1, self.name, self.token)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

# 使用
with DistributedLock(client, "resource"):
    # 临界区
    ...
```

## 6. 限流

```python
class RateLimiter:
    """固定窗口限流"""

    def __init__(self, client, name, max_requests=100, window=60):
        self.client = client
        self.name = name
        self.max_requests = max_requests
        self.window = window

    def is_allowed(self, identifier: str) -> bool:
        key = f"ratelimit:{self.name}:{identifier}"

        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.window)
        count, _ = pipe.execute()

        return count <= self.max_requests

# 使用
limiter = RateLimiter(client, "api", max_requests=100, window=60)

if limiter.is_allowed("user:123"):
    # 允许请求
    ...
else:
    raise RateLimitExceeded()
```

## 7. 管道（Pipeline）

```python
# 批量执行命令，减少网络往返
pipe = client.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.get("key1")
pipe.get("key2")
results = pipe.execute()
# [True, True, b"value1", b"value2"]
```

## Python vs JavaScript 对比

| 操作 | redis-py | ioredis (JS) |
|------|----------|--------------|
| 连接 | `redis.Redis()` | `new Redis()` |
| 设置 | `client.set()` | `client.set()` |
| 获取 | `client.get()` | `await client.get()` |
| 过期 | `client.setex()` | `client.setex()` |
| 管道 | `client.pipeline()` | `client.pipeline()` |


