# 数据库高级话题

> N+1 问题、事务隔离、连接池、锁

## N+1 问题

### 什么是 N+1

```python
# 获取所有用户
users = session.query(User).all()  # 1 次查询

# 访问每个用户的文章
for user in users:
    print(user.posts)  # N 次查询！

# 总共: 1 + N 次查询
```

### 解决方案

#### 1. joinedload（JOIN 查询）

```python
from sqlalchemy.orm import joinedload

# 一次 JOIN 查询
users = await session.execute(
    select(User).options(joinedload(User.posts))
)

# SQL: SELECT ... FROM users LEFT JOIN posts ON ...
```

**适用场景**：多对一、一对一关系

#### 2. selectinload（单独 IN 查询）

```python
from sqlalchemy.orm import selectinload

# 两次查询
users = await session.execute(
    select(User).options(selectinload(User.posts))
)

# SQL 1: SELECT * FROM users
# SQL 2: SELECT * FROM posts WHERE author_id IN (1, 2, 3, ...)
```

**适用场景**：一对多关系（推荐）

#### 3. subqueryload（子查询）

```python
from sqlalchemy.orm import subqueryload

users = await session.execute(
    select(User).options(subqueryload(User.posts))
)

# SQL: SELECT ... FROM posts WHERE author_id IN (SELECT id FROM users)
```

#### 选择指南

| 方法 | 查询数 | 适用场景 |
|------|--------|---------|
| `joinedload` | 1 | 多对一、数据量小 |
| `selectinload` | 2 | 一对多（推荐）|
| `subqueryload` | 2 | 复杂过滤 |

#### 嵌套预加载

```python
# 用户 → 文章 → 评论
users = await session.execute(
    select(User)
    .options(
        selectinload(User.posts)
        .selectinload(Post.comments)
    )
)
```

---

## 事务隔离级别

### 隔离级别对比

| 级别 | 脏读 | 不可重复读 | 幻读 |
|------|------|-----------|------|
| READ UNCOMMITTED | ✗ | ✗ | ✗ |
| READ COMMITTED | ✓ | ✗ | ✗ |
| REPEATABLE READ | ✓ | ✓ | ✗ |
| SERIALIZABLE | ✓ | ✓ | ✓ |

### 设置隔离级别

```python
# 全局设置
engine = create_async_engine(
    DATABASE_URL,
    isolation_level="REPEATABLE READ"
)

# 单次事务设置
async with session.begin():
    await session.execute(
        text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
    )
    # 执行操作...
```

### 常见问题

#### 脏读

```
事务 A: 写入数据（未提交）
事务 B: 读取到未提交的数据
事务 A: 回滚
事务 B: 读到的数据是脏的
```

#### 不可重复读

```
事务 A: 读取数据（值为 100）
事务 B: 修改数据为 200 并提交
事务 A: 再次读取（值为 200，不一致！）
```

#### 幻读

```
事务 A: 查询条件数据（5 条）
事务 B: 插入符合条件的数据并提交
事务 A: 再次查询（6 条，多了！）
```

---

## 连接池调优

### 配置参数

```python
engine = create_async_engine(
    DATABASE_URL,

    # 核心参数
    pool_size=10,          # 常驻连接数（默认 5）
    max_overflow=20,       # 超出时最大额外连接（默认 10）

    # 超时参数
    pool_timeout=30,       # 获取连接超时秒数
    pool_recycle=1800,     # 连接最大存活秒数（防止数据库端超时）

    # 健康检查
    pool_pre_ping=True,    # 使用前 ping 检查连接

    # 调试
    echo=False,            # 不打印 SQL
    echo_pool="debug",     # 打印连接池事件
)
```

### 计算连接数

```
推荐公式：
pool_size = (CPU 核数 * 2) + 硬盘数

示例（4 核 + 1 SSD）：
pool_size = (4 * 2) + 1 = 9

高并发场景：
pool_size = 20
max_overflow = 30
```

### 监控连接池

```python
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    print(f"Connection checked out: {connection_record}")

@event.listens_for(engine.sync_engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    print(f"Connection returned: {connection_record}")

# 获取池状态
def get_pool_status():
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
    }
```

---

## 软删除

### 模型实现

```python
from sqlalchemy import Column, Boolean, DateTime
from datetime import datetime

class SoftDeleteMixin:
    """软删除混入类"""
    is_deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)

    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

class User(Base, SoftDeleteMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
```

### 自动过滤

```python
from sqlalchemy.orm import Query

class SoftDeleteQuery(Query):
    """自动过滤已删除记录的查询"""

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._with_deleted = kwargs.pop('_with_deleted', False)
        return obj

    def __init__(self, *args, **kwargs):
        kwargs.pop('_with_deleted', None)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if not self._with_deleted:
            self._filter_deleted()
        return super().__iter__()

    def _filter_deleted(self):
        for mapper in self.column_descriptions:
            entity = mapper['entity']
            if hasattr(entity, 'is_deleted'):
                self._query = self.filter(entity.is_deleted == False)

    def with_deleted(self):
        """包含已删除记录"""
        return self.__class__(
            self._entity_from_pre_ent_zero().entity_zero,
            session=self.session,
            _with_deleted=True
        )
```

### 使用示例

```python
# 软删除
async def soft_delete_user(user_id: int):
    async with async_session() as session:
        user = await session.get(User, user_id)
        if user:
            user.soft_delete()
            await session.commit()

# 查询（自动排除已删除）
async def get_active_users():
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.is_deleted == False)
        )
        return list(result.scalars().all())

# 恢复
async def restore_user(user_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user:
            user.is_deleted = False
            user.deleted_at = None
            await session.commit()
```

---

## 乐观锁与悲观锁

### 乐观锁

假设冲突很少，使用版本号检测冲突。

```python
from sqlalchemy import Column, Integer
from sqlalchemy.orm import validates

class OptimisticLockMixin:
    version = Column(Integer, default=1, nullable=False)

class Product(Base, OptimisticLockMixin):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    stock = Column(Integer, default=0)

# 使用
async def update_stock_optimistic(product_id: int, quantity: int):
    async with async_session() as session:
        # 获取当前版本
        product = await session.get(Product, product_id)
        current_version = product.version

        # 更新时检查版本
        result = await session.execute(
            update(Product)
            .where(
                Product.id == product_id,
                Product.version == current_version
            )
            .values(
                stock=Product.stock - quantity,
                version=Product.version + 1
            )
        )

        if result.rowcount == 0:
            raise ConcurrencyError("Data was modified by another transaction")

        await session.commit()
```

### 悲观锁

假设冲突频繁，直接锁定行。

```python
from sqlalchemy import select

async def update_stock_pessimistic(product_id: int, quantity: int):
    async with async_session() as session:
        async with session.begin():
            # FOR UPDATE 锁定行
            result = await session.execute(
                select(Product)
                .where(Product.id == product_id)
                .with_for_update()  # 悲观锁
            )
            product = result.scalar_one()

            if product.stock < quantity:
                raise ValueError("Insufficient stock")

            product.stock -= quantity
            # 事务结束自动释放锁
```

### 锁选择

| 场景 | 推荐 |
|------|------|
| 读多写少 | 乐观锁 |
| 写频繁 | 悲观锁 |
| 短事务 | 乐观锁 |
| 长事务 | 悲观锁（注意超时）|
| 库存扣减 | 悲观锁 |
| 数据编辑 | 乐观锁 |

---

## 复杂查询优化

### 使用索引

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True)
    name = Column(String(100), index=True)
    created_at = Column(DateTime, index=True)

    # 复合索引
    __table_args__ = (
        Index('ix_user_name_created', 'name', 'created_at'),
    )
```

### EXPLAIN 分析

```python
async def analyze_query():
    async with async_session() as session:
        # 获取执行计划
        result = await session.execute(
            text("EXPLAIN ANALYZE SELECT * FROM users WHERE name = 'test'")
        )
        for row in result:
            print(row)
```

### 分页优化

```python
# 错误：OFFSET 大时性能差
async def get_users_bad(page: int, size: int):
    return await session.execute(
        select(User)
        .offset((page - 1) * size)  # 需要扫描跳过的行
        .limit(size)
    )

# 正确：基于游标分页
async def get_users_good(last_id: int | None, size: int):
    query = select(User).order_by(User.id).limit(size)
    if last_id:
        query = query.where(User.id > last_id)
    return await session.execute(query)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| N+1 查询 | 循环中访问关系 | 使用 selectinload |
| 大 OFFSET | 性能随页数下降 | 游标分页 |
| 忘记索引 | 全表扫描 | WHERE/ORDER 字段加索引 |
| 长事务 | 锁竞争严重 | 尽快提交 |
| SELECT * | 获取不需要的列 | 指定需要的列 |

---

## 小结

1. **N+1**：使用 selectinload/joinedload 预加载
2. **隔离级别**：根据需求选择，默认 READ COMMITTED
3. **连接池**：合理配置 pool_size 和超时
4. **软删除**：is_deleted 字段 + 过滤查询
5. **锁**：乐观锁适合读多写少，悲观锁适合写频繁

