# SQLAlchemy 异步支持

> 使用 async/await 进行数据库操作

## 为什么需要异步数据库

```
同步方式：
请求 → 查询数据库（等待...） → 返回响应
         ↑ 线程阻塞

异步方式：
请求 → 发起查询 → 处理其他请求 → 查询完成 → 返回响应
                 ↑ 不阻塞，高并发
```

---

## 安装依赖

```bash
# 异步 PostgreSQL
pip install asyncpg sqlalchemy[asyncio]

# 异步 MySQL
pip install aiomysql sqlalchemy[asyncio]

# 异步 SQLite（测试用）
pip install aiosqlite sqlalchemy[asyncio]
```

---

## 配置异步引擎

### 基础配置

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 注意：使用异步驱动 URL
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/db"
# MySQL: "mysql+aiomysql://user:pass@localhost/db"
# SQLite: "sqlite+aiosqlite:///./test.db"

# 创建异步引擎
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # 打印 SQL
    pool_size=5,
    max_overflow=10,
)

# 创建异步 Session 工厂
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 推荐：避免延迟加载问题
)

Base = declarative_base()
```

### 模型定义（与同步相同）

```python
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    posts = relationship("Post", back_populates="author", lazy="selectin")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))
    content = Column(String(10000))
    author_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    author = relationship("User", back_populates="posts")
```

---

## 基础 CRUD 操作

### 创建

```python
from sqlalchemy import select

async def create_user(email: str, name: str) -> User:
    async with async_session() as session:
        user = User(email=email, name=name)
        session.add(user)
        await session.commit()
        await session.refresh(user)  # 获取生成的 ID
        return user
```

### 查询

```python
async def get_user(user_id: int) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

async def get_users(skip: int = 0, limit: int = 100) -> list[User]:
    async with async_session() as session:
        result = await session.execute(
            select(User).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

async def get_user_by_email(email: str) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
```

### 更新

```python
async def update_user(user_id: int, name: str) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user:
            user.name = name
            await session.commit()
            await session.refresh(user)

        return user

# 批量更新
from sqlalchemy import update

async def deactivate_old_users(days: int = 365):
    async with async_session() as session:
        cutoff = datetime.utcnow() - timedelta(days=days)
        await session.execute(
            update(User)
            .where(User.last_login < cutoff)
            .values(is_active=False)
        )
        await session.commit()
```

### 删除

```python
from sqlalchemy import delete

async def delete_user(user_id: int) -> bool:
    async with async_session() as session:
        result = await session.execute(
            delete(User).where(User.id == user_id)
        )
        await session.commit()
        return result.rowcount > 0
```

---

## 与 FastAPI 集成

### 依赖注入

```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

# 依赖：获取 Session
async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# 使用
@app.get("/users/{user_id}")
async def read_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404)
    return user
```

### 生命周期管理

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # 关闭时清理
    await engine.dispose()

app = FastAPI(lifespan=lifespan)
```

---

## 关系查询

### 预加载（Eager Loading）

```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload: 单独查询（推荐用于一对多）
async def get_user_with_posts(user_id: int) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User)
            .options(selectinload(User.posts))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

# joinedload: JOIN 查询（推荐用于多对一）
async def get_posts_with_author() -> list[Post]:
    async with async_session() as session:
        result = await session.execute(
            select(Post)
            .options(joinedload(Post.author))
        )
        return list(result.scalars().all())
```

### lazy 属性设置

```python
class User(Base):
    __tablename__ = "users"
    # ...

    # 异步环境推荐使用 selectin 或 joined
    posts = relationship(
        "Post",
        back_populates="author",
        lazy="selectin"  # 自动预加载
    )
```

---

## 事务管理

### 基础事务

```python
async def transfer_points(from_id: int, to_id: int, points: int):
    async with async_session() as session:
        async with session.begin():  # 自动提交/回滚
            # 获取两个用户
            from_user = await session.get(User, from_id)
            to_user = await session.get(User, to_id)

            if not from_user or not to_user:
                raise ValueError("User not found")

            if from_user.points < points:
                raise ValueError("Insufficient points")

            from_user.points -= points
            to_user.points += points
            # begin() 块结束时自动提交
```

### 手动事务控制

```python
async def complex_operation():
    async with async_session() as session:
        try:
            # 操作 1
            user = User(email="test@example.com", name="Test")
            session.add(user)

            # 可以设置保存点
            savepoint = await session.begin_nested()

            try:
                # 操作 2（可能失败）
                risky_operation()
            except Exception:
                await savepoint.rollback()
                # 继续其他操作

            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

---

## 复杂查询

### 聚合查询

```python
from sqlalchemy import func

async def get_user_stats() -> dict:
    async with async_session() as session:
        # 用户总数
        total = await session.execute(
            select(func.count(User.id))
        )

        # 按月统计注册
        monthly = await session.execute(
            select(
                func.date_trunc('month', User.created_at).label('month'),
                func.count(User.id).label('count')
            )
            .group_by('month')
            .order_by('month')
        )

        return {
            "total": total.scalar(),
            "monthly": [
                {"month": row.month, "count": row.count}
                for row in monthly
            ]
        }
```

### 子查询

```python
async def get_users_with_post_count():
    async with async_session() as session:
        # 子查询：每个用户的文章数
        post_count = (
            select(
                Post.author_id,
                func.count(Post.id).label('post_count')
            )
            .group_by(Post.author_id)
            .subquery()
        )

        # 主查询
        result = await session.execute(
            select(
                User,
                func.coalesce(post_count.c.post_count, 0).label('post_count')
            )
            .outerjoin(post_count, User.id == post_count.c.author_id)
        )

        return [
            {"user": user, "post_count": count}
            for user, count in result
        ]
```

---

## 性能优化

### 连接池配置

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,          # 常驻连接数
    max_overflow=20,       # 超出时最大额外连接
    pool_timeout=30,       # 获取连接超时（秒）
    pool_recycle=1800,     # 连接回收时间（秒）
    pool_pre_ping=True,    # 使用前检查连接
)
```

### 批量操作

```python
# 批量插入
async def bulk_create_users(users_data: list[dict]):
    async with async_session() as session:
        users = [User(**data) for data in users_data]
        session.add_all(users)
        await session.commit()

# 使用 execute 批量插入（更快）
from sqlalchemy import insert

async def bulk_insert_users(users_data: list[dict]):
    async with async_session() as session:
        await session.execute(
            insert(User),
            users_data
        )
        await session.commit()
```

### 流式查询

```python
async def stream_large_data():
    async with async_session() as session:
        # 流式获取，不一次加载所有
        result = await session.stream(
            select(User).order_by(User.id)
        )

        async for user in result.scalars():
            yield user
```

---

## 与同步代码的区别

| 同步 | 异步 | 说明 |
|------|------|------|
| `create_engine()` | `create_async_engine()` | 创建引擎 |
| `Session` | `AsyncSession` | Session 类型 |
| `session.query()` | `select()` + `await session.execute()` | 查询方式 |
| `session.commit()` | `await session.commit()` | 提交 |
| `session.refresh()` | `await session.refresh()` | 刷新 |
| `relationship(lazy="select")` | `relationship(lazy="selectin")` | 延迟加载 |

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 使用同步驱动 | 阻塞事件循环 | 使用 asyncpg/aiomysql |
| lazy="select" | 异步环境会报错 | 使用 selectin/joined |
| 忘记 await | 没有执行 | 所有异步方法加 await |
| expire_on_commit=True | commit 后属性失效 | 设为 False |
| session.query() | 已弃用 | 使用 select() |

---

## 小结

1. **驱动**：asyncpg(PostgreSQL)、aiomysql(MySQL)
2. **Session**：`AsyncSession` + `async_session()`
3. **查询**：`select()` + `await session.execute()`
4. **关系**：使用 `selectinload` / `joinedload`
5. **事务**：`async with session.begin()`

