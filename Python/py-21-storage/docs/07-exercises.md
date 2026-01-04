# 练习题

## SQLAlchemy 练习

### 练习 1: 模型定义

定义一个博客系统的模型：

```python
# 要求:
# 1. User 模型：id, username, email, created_at
# 2. Post 模型：id, title, content, author_id, created_at, published
# 3. Comment 模型：id, content, post_id, author_id, created_at
# 4. User 和 Post 是一对多关系
# 5. Post 和 Comment 是一对多关系
# 6. User 和 Comment 是一对多关系

# TODO: 实现模型
```

### 练习 2: CRUD 操作

实现文章的 CRUD 操作：

```python
# 要求:
# 1. create_post(session, title, content, author_id) -> Post
# 2. get_post(session, post_id) -> Post | None
# 3. get_posts_by_author(session, author_id) -> list[Post]
# 4. update_post(session, post_id, title, content) -> Post
# 5. delete_post(session, post_id) -> bool
# 6. publish_post(session, post_id) -> Post

# TODO: 实现函数
```

### 练习 3: 复杂查询

实现以下查询：

```python
# 要求:
# 1. 获取发布文章数量最多的 5 个用户
# 2. 获取最近 7 天发布的文章
# 3. 获取评论数最多的 10 篇文章
# 4. 搜索标题或内容包含关键词的文章

# TODO: 实现查询
```

### 练习 4: N+1 问题

修复以下 N+1 问题：

```python
# 问题代码
def get_posts_with_comments(session):
    posts = session.query(Post).all()
    for post in posts:
        print(post.title)
        for comment in post.comments:  # N+1!
            print(f"  - {comment.content}")

# TODO: 使用预加载修复
```

## Repository 练习

### 练习 5: 实现 Repository

为博客系统实现 Repository：

```python
# 要求:
# 1. PostRepository 继承 BaseRepository
# 2. 实现 get_published() 方法
# 3. 实现 get_by_author(author_id) 方法
# 4. 实现 search(keyword) 方法
# 5. 实现 get_with_comments(post_id) 方法

# TODO: 实现 Repository
```

### 练习 6: 依赖注入

创建 FastAPI 端点使用 Repository：

```python
# 要求:
# 1. GET /posts - 获取所有已发布文章
# 2. GET /posts/{id} - 获取文章详情（包含评论）
# 3. POST /posts - 创建文章
# 4. PUT /posts/{id} - 更新文章
# 5. 使用依赖注入获取 Repository

# TODO: 实现端点
```

## Redis 练习

### 练习 7: 缓存实现

实现文章缓存：

```python
# 要求:
# 1. 实现 get_post_cached(post_id) 函数
# 2. 先查缓存，命中返回
# 3. 未命中查数据库，写入缓存
# 4. 缓存 TTL 为 5 分钟
# 5. 更新文章时清除缓存

# TODO: 实现缓存
```

### 练习 8: 排行榜

使用 Redis Sorted Set 实现文章热度排行：

```python
# 要求:
# 1. record_view(post_id) - 记录文章浏览
# 2. record_like(post_id) - 记录点赞（权重更高）
# 3. get_hot_posts(limit=10) - 获取热门文章
# 4. 热度 = 浏览数 + 点赞数 * 5
# 5. 每小时重置排行榜

# TODO: 实现排行榜
```

### 练习 9: 分布式锁

使用分布式锁防止重复操作：

```python
# 要求:
# 1. 实现 DistributedLock 类
# 2. acquire() 获取锁
# 3. release() 释放锁
# 4. 支持超时自动释放
# 5. 防止误删其他进程的锁

# TODO: 实现分布式锁
```

### 练习 10: 限流

实现 API 限流：

```python
# 要求:
# 1. 实现固定窗口限流
# 2. 每分钟最多 100 次请求
# 3. 支持不同用户独立计数
# 4. 返回剩余请求次数

# TODO: 实现限流
```

## 任务队列练习

### 练习 11: 简单队列

实现邮件发送队列：

```python
# 要求:
# 1. 入队：enqueue_email(to, subject, body)
# 2. 出队：get_pending_email()
# 3. 完成：mark_completed(task_id)
# 4. 失败：mark_failed(task_id, error)
# 5. 支持重试（最多 3 次）

# TODO: 实现队列
```

### 练习 12: Worker

实现任务 Worker：

```python
# 要求:
# 1. 从队列获取任务
# 2. 根据任务类型调用不同处理函数
# 3. 处理成功标记完成
# 4. 处理失败标记失败并重试
# 5. 支持优雅停止

# TODO: 实现 Worker
```

## 参考答案

### 练习 4 答案

```python
from sqlalchemy.orm import selectinload

def get_posts_with_comments(session):
    # 使用预加载
    stmt = select(Post).options(selectinload(Post.comments))
    posts = session.execute(stmt).scalars().all()

    for post in posts:
        print(post.title)
        for comment in post.comments:  # 不会产生额外查询
            print(f"  - {comment.content}")
```

### 练习 7 答案

```python
import json

def get_post_cached(post_id: int):
    cache_key = f"post:{post_id}"

    # 1. 查缓存
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. 查数据库
    post = session.get(Post, post_id)
    if post is None:
        return None

    # 3. 写缓存
    post_data = {
        "id": post.id,
        "title": post.title,
        "content": post.content,
    }
    redis_client.setex(cache_key, 300, json.dumps(post_data))

    return post_data

def invalidate_post_cache(post_id: int):
    cache_key = f"post:{post_id}"
    redis_client.delete(cache_key)
```

### 练习 9 答案

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
        lua = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        return self.client.eval(lua, 1, self.name, self.token)

    def __enter__(self):
        if not self.acquire():
            raise Exception("Failed to acquire lock")
        return self

    def __exit__(self, *args):
        self.release()
```


