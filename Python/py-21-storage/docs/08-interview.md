# 面试高频题

## 1. SQLAlchemy 的 Session 是什么？

### 答案

Session 是 SQLAlchemy 的工作单元（Unit of Work），负责：

1. **对象追踪** - 追踪加载和修改的对象
2. **事务管理** - 管理数据库事务
3. **标识映射** - 确保同一主键只有一个对象实例
4. **持久化** - 将变更同步到数据库

```python
from sqlalchemy.orm import Session

# Session 生命周期
session = Session(engine)

# 1. 加载对象 - 对象处于 "persistent" 状态
user = session.get(User, 1)

# 2. 修改对象 - 变更被追踪
user.name = "New Name"

# 3. 提交 - 变更写入数据库
session.commit()

# 4. 关闭 - 释放资源
session.close()
```

**状态**：
- **Transient** - 新建，未加入 Session
- **Pending** - 已加入 Session，未提交
- **Persistent** - 已提交，Session 追踪中
- **Detached** - Session 关闭后

---

## 2. 如何处理 N+1 查询问题？

### 答案

**N+1 问题**：查询主表 1 次 + 每条记录查询关联表 N 次

**解决方案**：

1. **selectinload** - 使用 IN 子查询（推荐）

```python
from sqlalchemy.orm import selectinload

stmt = select(User).options(selectinload(User.items))
users = session.execute(stmt).scalars().all()
# 2 次查询：SELECT users... + SELECT items WHERE user_id IN (...)
```

2. **joinedload** - 使用 JOIN

```python
from sqlalchemy.orm import joinedload

stmt = select(User).options(joinedload(User.profile))
# 1 次查询：SELECT users... LEFT JOIN profiles...
```

3. **subqueryload** - 使用子查询

```python
from sqlalchemy.orm import subqueryload

stmt = select(User).options(subqueryload(User.items))
```

**选择策略**：
- 一对多/多对多：`selectinload`
- 一对一/多对一：`joinedload`
- 大数据集：`subqueryload`

---

## 3. 缓存穿透、缓存雪崩是什么？

### 答案

**缓存穿透**：查询不存在的数据，每次都穿透到数据库

```python
# 解决：缓存空值
def get_user(user_id):
    cached = redis.get(f"user:{user_id}")
    if cached == "NULL":
        return None  # 命中空值
    if cached:
        return json.loads(cached)
    
    user = db.get_user(user_id)
    if user:
        redis.setex(f"user:{user_id}", 300, json.dumps(user))
    else:
        redis.setex(f"user:{user_id}", 60, "NULL")  # 缓存空值
    return user
```

**缓存雪崩**：大量缓存同时过期

```python
# 解决：过期时间加随机抖动
import random

def cache_with_jitter(key, value, base_ttl):
    jitter = random.randint(0, 60)
    redis.setex(key, base_ttl + jitter, value)
```

**缓存击穿**：热点数据过期瞬间大量请求

```python
# 解决：分布式锁
def get_hot_data(key):
    cached = redis.get(key)
    if cached:
        return cached
    
    if redis.set(f"lock:{key}", "1", nx=True, ex=10):
        try:
            data = db.get(key)
            redis.setex(key, 300, data)
            return data
        finally:
            redis.delete(f"lock:{key}")
    else:
        time.sleep(0.1)
        return get_hot_data(key)
```

---

## 4. Redis 的数据类型有哪些？

### 答案

| 类型 | 描述 | 使用场景 |
|------|------|----------|
| **String** | 字符串 | 缓存、计数器、分布式锁 |
| **Hash** | 哈希表 | 对象存储、用户信息 |
| **List** | 列表 | 消息队列、时间线 |
| **Set** | 集合 | 标签、好友关系 |
| **Sorted Set** | 有序集合 | 排行榜、延时队列 |
| **Stream** | 流 | 消息队列（高级） |
| **HyperLogLog** | 基数统计 | UV 统计 |
| **Bitmap** | 位图 | 签到、在线状态 |

```python
# String
redis.set("name", "John")
redis.incr("counter")

# Hash
redis.hset("user:1", mapping={"name": "John", "age": 30})

# List
redis.lpush("queue", "task1")
redis.rpop("queue")

# Set
redis.sadd("tags", "python", "redis")

# Sorted Set
redis.zadd("leaderboard", {"player1": 100})
```

---

## 5. 如何实现分布式锁？

### 答案

```python
import uuid

class DistributedLock:
    def __init__(self, client, name, timeout=10):
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = timeout
        self.token = str(uuid.uuid4())
    
    def acquire(self):
        # SET key value NX EX timeout
        return self.client.set(
            self.name,
            self.token,
            nx=True,  # 只在不存在时设置
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
```

**关键点**：
1. **唯一标识** - 防止误删其他进程的锁
2. **原子性** - 使用 Lua 脚本
3. **超时** - 防止死锁
4. **续期** - 长任务需要续期（Redlock）

---

## 6. 数据库迁移的最佳实践？

### 答案

1. **版本控制**
```bash
# 每次变更生成迁移脚本
alembic revision --autogenerate -m "add user table"
```

2. **先测试后执行**
```bash
# 生成 SQL 预览
alembic upgrade head --sql
```

3. **备份**
```bash
# 迁移前备份
pg_dump mydb > backup_$(date +%Y%m%d).sql
```

4. **小步迭代**
```python
# 分步执行大变更
# 1. 添加新列（可为空）
# 2. 迁移数据
# 3. 修改约束
# 4. 删除旧列
```

5. **向后兼容**
```python
# 新增列设置默认值
op.add_column('users', sa.Column('status', sa.String(20), server_default='active'))
```

---

## 7. 如何处理数据库连接池？

### 答案

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://...",
    pool_size=5,        # 连接池大小
    max_overflow=10,    # 最大溢出连接数
    pool_timeout=30,    # 获取连接超时
    pool_recycle=1800,  # 连接回收时间（秒）
    pool_pre_ping=True, # 获取前检查连接是否有效
)
```

**关键配置**：
- **pool_size** - 常驻连接数
- **max_overflow** - 峰值时额外连接
- **pool_timeout** - 获取连接超时
- **pool_recycle** - 定期回收防止连接失效
- **pool_pre_ping** - 使用前检查连接

**监控**：
```python
from sqlalchemy import event

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    # 连接被取出
    pass

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    # 连接被归还
    pass
```

---

## 8. 什么时候用缓存？

### 答案

**适合缓存**：
1. 读多写少的数据
2. 计算耗时的结果
3. 访问频率高的数据
4. 对一致性要求不高的数据

**不适合缓存**：
1. 写多读少的数据
2. 强一致性要求的数据
3. 数据量太大的内容
4. 随机访问的数据

**缓存策略选择**：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| Cache-Aside | 应用管理缓存 | 通用场景 |
| Read-Through | 缓存负责读取 | 简化应用逻辑 |
| Write-Through | 同步写缓存和DB | 数据一致性要求高 |
| Write-Behind | 异步写DB | 写入性能要求高 |

```python
# Cache-Aside 示例
def get_user(user_id):
    # 1. 查缓存
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    
    # 2. 查数据库
    user = db.get_user(user_id)
    
    # 3. 写缓存
    if user:
        cache.set(f"user:{user_id}", user, ttl=300)
    
    return user
```

---

## 附加题

### 9. ORM 和原生 SQL 如何选择？

**使用 ORM**：
- 简单 CRUD 操作
- 需要对象映射
- 需要关系处理
- 代码可维护性优先

**使用原生 SQL**：
- 复杂查询和报表
- 性能关键路径
- 批量操作
- 数据库特定功能

```python
# 混合使用
from sqlalchemy import text

# 复杂查询用原生 SQL
result = session.execute(
    text("SELECT * FROM users WHERE age > :age"),
    {"age": 18}
)

# 简单操作用 ORM
user = session.get(User, 1)
```

### 10. 如何实现乐观锁？

```python
class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    quantity = Column(Integer)
    version = Column(Integer, default=0)  # 版本号

def update_quantity(session, item_id, new_quantity):
    item = session.get(Item, item_id)
    old_version = item.version
    
    # 更新时检查版本
    result = session.execute(
        text("""
            UPDATE items
            SET quantity = :quantity, version = version + 1
            WHERE id = :id AND version = :version
        """),
        {"quantity": new_quantity, "id": item_id, "version": old_version}
    )
    
    if result.rowcount == 0:
        raise ConcurrencyError("Data was modified by another transaction")
    
    session.commit()
```


