# 关系与查询

## 1. 一对多关系

### 1.1 定义

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, Mapped

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50))

    # 一对多：一个用户有多个商品
    items: Mapped[list["Item"]] = relationship(
        "Item",
        back_populates="owner",
        cascade="all, delete-orphan",  # 级联删除
    )

class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

    # 外键
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # 多对一
    owner: Mapped["User"] = relationship("User", back_populates="items")
```

### 1.2 使用

```python
# 创建用户和商品
user = User(username="john")
item1 = Item(name="Item 1", owner=user)
item2 = Item(name="Item 2", owner=user)

session.add(user)
session.commit()

# 访问关系
print(user.items)  # [<Item 1>, <Item 2>]
print(item1.owner.username)  # "john"
```

## 2. 多对多关系

### 2.1 定义

```python
from sqlalchemy import Table, Column, Integer, ForeignKey

# 关联表
item_tags = Table(
    "item_tags",
    Base.metadata,
    Column("item_id", Integer, ForeignKey("items.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

    # 多对多
    tags: Mapped[list["Tag"]] = relationship(
        "Tag",
        secondary=item_tags,
        back_populates="items",
    )

class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)

    # 反向关系
    items: Mapped[list["Item"]] = relationship(
        "Item",
        secondary=item_tags,
        back_populates="tags",
    )
```

### 2.2 使用

```python
# 创建
tag1 = Tag(name="electronics")
tag2 = Tag(name="sale")
item = Item(name="iPhone", tags=[tag1, tag2])

session.add(item)
session.commit()

# 访问
print(item.tags)  # [<Tag electronics>, <Tag sale>]
print(tag1.items)  # [<Item iPhone>]

# 添加标签
item.tags.append(Tag(name="new"))
session.commit()

# 移除标签
item.tags.remove(tag1)
session.commit()
```

## 3. 自引用关系

```python
class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

    # 父分类 ID
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("categories.id"),
        nullable=True
    )

    # 自引用
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        remote_side=[id],
        back_populates="children",
    )

    children: Mapped[list["Category"]] = relationship(
        "Category",
        back_populates="parent",
    )
```

## 4. 加载策略

### 4.1 懒加载（默认）

```python
# 访问时才查询
user = session.get(User, 1)
print(user.items)  # 此时执行 SQL 查询
```

### 4.2 预加载（Eager Loading）

```python
from sqlalchemy.orm import joinedload, selectinload

# joinedload - 使用 JOIN
stmt = select(User).options(joinedload(User.items)).where(User.id == 1)
user = session.execute(stmt).unique().scalar_one()

# selectinload - 使用 IN 子查询（推荐用于一对多）
stmt = select(User).options(selectinload(User.items))
users = session.execute(stmt).unique().scalars().all()
```

### 4.3 在模型中设置默认加载策略

```python
class User(Base):
    items: Mapped[list["Item"]] = relationship(
        "Item",
        back_populates="owner",
        lazy="selectin",  # 默认使用 selectin 加载
    )
```

## 5. N+1 问题

### 5.1 问题演示

```python
# 错误：N+1 查询
users = session.query(User).all()  # 1 次查询
for user in users:
    print(user.items)  # N 次查询（每个用户一次）
```

### 5.2 解决方案

```python
# 正确：使用预加载
stmt = select(User).options(selectinload(User.items))
users = session.execute(stmt).scalars().all()

for user in users:
    print(user.items)  # 不会产生额外查询
```

## 6. 联表查询

### 6.1 JOIN

```python
from sqlalchemy import select

# 显式 JOIN
stmt = (
    select(User, Item)
    .join(Item, User.id == Item.owner_id)
)

# 使用关系 JOIN
stmt = (
    select(User)
    .join(User.items)
    .where(Item.price > 100)
)

# LEFT JOIN
stmt = (
    select(User)
    .outerjoin(User.items)
)
```

### 6.2 子查询

```python
from sqlalchemy import func

# 子查询
subquery = (
    select(Item.owner_id, func.count(Item.id).label("item_count"))
    .group_by(Item.owner_id)
    .subquery()
)

# 主查询
stmt = (
    select(User, subquery.c.item_count)
    .outerjoin(subquery, User.id == subquery.c.owner_id)
)
```

## 7. 级联操作

### 7.1 级联类型

```python
class User(Base):
    items: Mapped[list["Item"]] = relationship(
        "Item",
        back_populates="owner",
        cascade="all, delete-orphan",
        # cascade 选项:
        # - "save-update": 添加到 session 时级联
        # - "merge": merge 时级联
        # - "expunge": 从 session 移除时级联
        # - "delete": 删除时级联
        # - "delete-orphan": 删除孤儿记录
        # - "all": 所有操作
    )
```

### 7.2 使用

```python
# 级联删除
user = session.get(User, 1)
session.delete(user)  # 自动删除关联的 items
session.commit()

# 删除孤儿
user.items.remove(item)  # item 变成孤儿
session.commit()  # item 被自动删除
```

## 8. 关系加载选项对比

| 策略 | SQL 查询数 | 适用场景 |
|------|-----------|----------|
| `lazy="select"` | N+1 | 很少访问关系 |
| `lazy="selectin"` | 2 | 一对多、多对多 |
| `lazy="joined"` | 1 | 一对一、多对一 |
| `lazy="subquery"` | 2 | 大数据集 |
| `lazy="dynamic"` | 按需 | 需要进一步过滤 |

## 9. 实战示例

```python
from sqlalchemy.orm import selectinload, joinedload

# 获取用户及其所有商品和商品标签
stmt = (
    select(User)
    .options(
        selectinload(User.items).selectinload(Item.tags)
    )
    .where(User.id == 1)
)
user = session.execute(stmt).unique().scalar_one()

# 现在可以访问所有数据，不会产生额外查询
for item in user.items:
    print(f"{item.name}: {[tag.name for tag in item.tags]}")
```


