# P21: 存储与缓存

> 掌握数据库操作、缓存策略、任务队列

## 🎯 学完后能做

- 使用 SQLAlchemy ORM
- 实现 Redis 缓存
- 理解任务队列

## 🚀 快速开始

```bash
# 进入项目目录
cd py-21-storage

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 初始化数据库（SQLite）
python -m storage_lab.cli db init

# 运行迁移
alembic upgrade head

# 启动 Redis（可选，需要 Docker）
docker run -d -p 6379:6379 redis:alpine
```

## 📁 目录结构

```
py-21-storage/
├── README.md
├── pyproject.toml
├── alembic.ini                  # Alembic 配置
├── docs/
│   ├── 01-sqlalchemy.md         # SQLAlchemy 基础
│   ├── 02-relationships.md      # 关系与查询
│   ├── 03-alembic.md            # 数据库迁移
│   ├── 04-repository.md         # Repository 模式
│   ├── 05-redis.md              # Redis 缓存
│   ├── 06-queue.md              # 任务队列
│   ├── 07-exercises.md          # 练习题
│   └── 08-interview.md          # 面试题
├── src/storage_lab/
│   ├── __init__.py
│   ├── cli.py                   # CLI 入口
│   ├── config.py                # 配置管理
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy 模型
│   │   ├── session.py           # 数据库会话
│   │   └── migrations/          # Alembic 迁移
│   │       ├── env.py
│   │       └── versions/
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── base.py              # 基础 Repository
│   │   ├── user_repo.py         # 用户 Repository
│   │   └── item_repo.py         # 商品 Repository
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── client.py            # Redis 客户端
│   │   ├── decorators.py        # 缓存装饰器
│   │   └── lock.py              # 分布式锁
│   └── queue/
│       ├── __init__.py
│       ├── simple.py            # 简单任务队列
│       └── worker.py            # Worker 实现
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_repositories.py
│   ├── test_cache.py
│   └── test_queue.py
└── scripts/
    ├── run_demo.sh
    └── test.sh
```

## 🔧 核心功能

### 1. SQLAlchemy ORM

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True)
    
    # 一对多关系
    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="items")
```

### 2. Repository 模式

```python
class UserRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)
    
    def get_by_email(self, email: str) -> User | None:
        return self.session.query(User).filter(User.email == email).first()
    
    def create(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.session.add(user)
        self.session.commit()
        return user
```

### 3. Redis 缓存

```python
import redis

class CacheClient:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.client = redis.from_url(url)
    
    def get(self, key: str) -> str | None:
        return self.client.get(key)
    
    def set(self, key: str, value: str, ttl: int = 300):
        self.client.setex(key, ttl, value)
    
    def delete(self, key: str):
        self.client.delete(key)
```

### 4. 分布式锁

```python
from contextlib import contextmanager

@contextmanager
def distributed_lock(client, lock_name: str, timeout: int = 10):
    lock = client.lock(lock_name, timeout=timeout)
    acquired = lock.acquire(blocking=True)
    try:
        if acquired:
            yield True
        else:
            yield False
    finally:
        if acquired:
            lock.release()
```

## 📚 学习路径

1. **SQLAlchemy** - 模型、关系、查询、事务
2. **Alembic** - 迁移脚本、升级降级
3. **Repository** - CRUD 抽象、依赖注入
4. **Redis** - 缓存策略、分布式锁
5. **任务队列** - 概念、简单实现

## ✅ 功能清单

- [x] SQLAlchemy 模型定义
- [x] 一对多、多对多关系
- [x] 查询 API
- [x] 事务处理
- [x] 异步支持
- [x] Alembic 迁移
- [x] Repository 模式
- [x] Redis 基础操作
- [x] 缓存策略（TTL）
- [x] 分布式锁
- [x] 限流
- [x] 任务队列概念


