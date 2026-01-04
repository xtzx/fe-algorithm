# SQLAlchemy 基础

## 概述

SQLAlchemy 是 Python 最流行的 ORM（对象关系映射）库，提供：

1. **Core** - SQL 表达式语言
2. **ORM** - 对象关系映射
3. **异步支持** - SQLAlchemy 2.0

## 1. 安装与配置

```bash
pip install sqlalchemy[asyncio]
pip install aiosqlite  # SQLite 异步驱动
pip install asyncpg    # PostgreSQL 异步驱动
```

### 数据库 URL 格式

```python
# SQLite
"sqlite:///./app.db"
"sqlite+aiosqlite:///./app.db"  # 异步

# PostgreSQL
"postgresql://user:password@host:port/database"
"postgresql+asyncpg://user:password@host:port/database"  # 异步

# MySQL
"mysql+pymysql://user:password@host:port/database"
"mysql+aiomysql://user:password@host:port/database"  # 异步
```

## 2. 模型定义

### 2.1 声明式基类

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, DateTime, func

class Base(DeclarativeBase):
    """所有模型的基类"""
    pass
```

### 2.2 基础模型

```python
from datetime import datetime
from typing import Optional

class User(Base):
    __tablename__ = "users"
    
    # 主键
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # 必填字段
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    
    # 可选字段
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # 默认值
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # 服务器默认值
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        server_default=func.now()
    )
    
    # 自动更新
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        onupdate=func.now(),
        nullable=True
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"
```

### 2.3 常用字段类型

```python
from sqlalchemy import (
    Integer,      # 整数
    BigInteger,   # 大整数
    String,       # 变长字符串
    Text,         # 长文本
    Boolean,      # 布尔
    DateTime,     # 日期时间
    Date,         # 日期
    Time,         # 时间
    Float,        # 浮点数
    DECIMAL,      # 精确小数
    JSON,         # JSON
    Enum,         # 枚举
    LargeBinary,  # 二进制
)
```

### 2.4 索引和约束

```python
from sqlalchemy import Index, UniqueConstraint, CheckConstraint

class Item(Base):
    __tablename__ = "items"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price: Mapped[Decimal] = mapped_column(DECIMAL(10, 2))
    
    # 表级配置
    __table_args__ = (
        # 索引
        Index("ix_items_name", "name"),
        
        # 联合唯一约束
        UniqueConstraint("name", "owner_id", name="uq_items_name_owner"),
        
        # 检查约束
        CheckConstraint("price >= 0", name="ck_items_price_positive"),
    )
```

## 3. 引擎和会话

### 3.1 创建引擎

```python
from sqlalchemy import create_engine

# 同步引擎
engine = create_engine(
    "sqlite:///./app.db",
    echo=True,           # 打印 SQL
    pool_size=5,         # 连接池大小
    max_overflow=10,     # 最大溢出连接数
    pool_pre_ping=True,  # 连接健康检查
)
```

### 3.2 创建会话

```python
from sqlalchemy.orm import sessionmaker, Session

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# 使用会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 3.3 异步引擎和会话

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# 异步引擎
async_engine = create_async_engine(
    "sqlite+aiosqlite:///./app.db",
    echo=True,
)

# 异步会话
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
)

# 使用
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
```

## 4. CRUD 操作

### 4.1 创建（Create）

```python
# 单条创建
user = User(username="john", email="john@example.com")
session.add(user)
session.commit()
session.refresh(user)  # 刷新获取自动生成的字段

# 批量创建
users = [
    User(username="user1", email="user1@example.com"),
    User(username="user2", email="user2@example.com"),
]
session.add_all(users)
session.commit()
```

### 4.2 查询（Read）

```python
from sqlalchemy import select

# 根据主键查询
user = session.get(User, 1)

# 条件查询
stmt = select(User).where(User.username == "john")
user = session.execute(stmt).scalar_one_or_none()

# 查询多条
stmt = select(User).where(User.is_active == True)
users = session.execute(stmt).scalars().all()

# 分页
stmt = select(User).offset(0).limit(10)
users = session.execute(stmt).scalars().all()

# 排序
stmt = select(User).order_by(User.created_at.desc())

# 计数
from sqlalchemy import func
stmt = select(func.count()).select_from(User)
count = session.execute(stmt).scalar()
```

### 4.3 更新（Update）

```python
# 方式1：获取后修改
user = session.get(User, 1)
user.email = "new@example.com"
session.commit()

# 方式2：批量更新
from sqlalchemy import update

stmt = (
    update(User)
    .where(User.is_active == False)
    .values(is_active=True)
)
session.execute(stmt)
session.commit()
```

### 4.4 删除（Delete）

```python
# 方式1：获取后删除
user = session.get(User, 1)
session.delete(user)
session.commit()

# 方式2：批量删除
from sqlalchemy import delete

stmt = delete(User).where(User.is_active == False)
session.execute(stmt)
session.commit()
```

## 5. 事务处理

### 5.1 基础事务

```python
try:
    user = User(username="test", email="test@example.com")
    session.add(user)
    
    # 其他操作...
    
    session.commit()  # 提交事务
except Exception:
    session.rollback()  # 回滚
    raise
```

### 5.2 嵌套事务（Savepoint）

```python
from sqlalchemy import savepoint

# 创建保存点
with session.begin_nested():
    try:
        user = User(username="test", email="test@example.com")
        session.add(user)
        # 可能失败的操作...
    except Exception:
        # 只回滚到保存点
        pass

# 继续其他操作
session.commit()
```

### 5.3 异步事务

```python
async with AsyncSessionLocal() as session:
    async with session.begin():
        user = User(username="test", email="test@example.com")
        session.add(user)
        # 自动提交或回滚
```

## 6. 查询进阶

### 6.1 过滤操作符

```python
from sqlalchemy import and_, or_, not_

# AND
stmt = select(User).where(
    and_(User.is_active == True, User.is_admin == False)
)

# OR
stmt = select(User).where(
    or_(User.username == "admin", User.is_admin == True)
)

# NOT
stmt = select(User).where(not_(User.is_active))

# IN
stmt = select(User).where(User.id.in_([1, 2, 3]))

# LIKE
stmt = select(User).where(User.username.like("%john%"))

# ILIKE（不区分大小写）
stmt = select(User).where(User.username.ilike("%john%"))

# BETWEEN
stmt = select(Item).where(Item.price.between(10, 100))

# IS NULL
stmt = select(User).where(User.full_name.is_(None))
```

### 6.2 聚合函数

```python
from sqlalchemy import func

# COUNT
stmt = select(func.count(User.id))

# SUM
stmt = select(func.sum(Item.price))

# AVG
stmt = select(func.avg(Item.price))

# GROUP BY
stmt = (
    select(Item.owner_id, func.count(Item.id))
    .group_by(Item.owner_id)
)

# HAVING
stmt = (
    select(Item.owner_id, func.count(Item.id).label("count"))
    .group_by(Item.owner_id)
    .having(func.count(Item.id) > 5)
)
```

## Python vs JavaScript 对比

| 概念 | SQLAlchemy | Prisma (JS) |
|------|------------|-------------|
| 模型 | Class 继承 | Schema 文件 |
| 查询 | `select()` | `findMany()` |
| 创建 | `session.add()` | `create()` |
| 更新 | 修改属性 | `update()` |
| 关系 | `relationship()` | `@relation` |
| 迁移 | Alembic | `prisma migrate` |


