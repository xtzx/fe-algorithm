# Alembic 数据库迁移

## 概述

Alembic 是 SQLAlchemy 的数据库迁移工具，提供：

1. 自动生成迁移脚本
2. 版本控制
3. 升级和降级
4. 多分支支持

## 1. 安装和初始化

```bash
# 安装
pip install alembic

# 初始化（在项目根目录）
alembic init alembic
```

### 目录结构

```
project/
├── alembic.ini         # 配置文件
└── alembic/
    ├── env.py          # 迁移环境
    ├── script.py.mako  # 脚本模板
    └── versions/       # 迁移脚本
```

## 2. 配置

### 2.1 alembic.ini

```ini
[alembic]
script_location = alembic
sqlalchemy.url = sqlite:///./app.db
```

### 2.2 env.py

```python
from alembic import context
from myapp.db.models import Base  # 导入模型基类

target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()
```

## 3. 生成迁移

### 3.1 自动生成

```bash
# 检测模型变化，自动生成迁移脚本
alembic revision --autogenerate -m "add user table"
```

生成的脚本：

```python
"""add user table

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2024-01-01 12:00:00

"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = None

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(200), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
    )

def downgrade():
    op.drop_table('users')
```

### 3.2 手动编写

```bash
# 创建空迁移脚本
alembic revision -m "add index"
```

```python
def upgrade():
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    op.drop_index('ix_users_email', 'users')
```

## 4. 执行迁移

```bash
# 升级到最新版本
alembic upgrade head

# 升级到指定版本
alembic upgrade a1b2c3d4e5f6

# 升级一个版本
alembic upgrade +1

# 降级一个版本
alembic downgrade -1

# 降级到初始状态
alembic downgrade base

# 查看当前版本
alembic current

# 查看历史
alembic history
```

## 5. 常用操作

### 5.1 添加列

```python
def upgrade():
    op.add_column('users', sa.Column('phone', sa.String(20), nullable=True))

def downgrade():
    op.drop_column('users', 'phone')
```

### 5.2 修改列

```python
def upgrade():
    op.alter_column(
        'users',
        'username',
        existing_type=sa.String(50),
        type_=sa.String(100),
    )

def downgrade():
    op.alter_column(
        'users',
        'username',
        existing_type=sa.String(100),
        type_=sa.String(50),
    )
```

### 5.3 添加索引

```python
def upgrade():
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_items_name_price', 'items', ['name', 'price'])

def downgrade():
    op.drop_index('ix_items_name_price', 'items')
    op.drop_index('ix_users_email', 'users')
```

### 5.4 添加外键

```python
def upgrade():
    op.add_column('items', sa.Column('owner_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_items_owner',
        'items', 'users',
        ['owner_id'], ['id'],
    )

def downgrade():
    op.drop_constraint('fk_items_owner', 'items', type_='foreignkey')
    op.drop_column('items', 'owner_id')
```

### 5.5 数据迁移

```python
from alembic import op
from sqlalchemy import orm
from sqlalchemy.sql import table, column

def upgrade():
    # 创建临时表引用
    users = table('users',
        column('id', sa.Integer),
        column('status', sa.String),
    )

    # 更新数据
    op.execute(
        users.update()
        .where(users.c.status == None)
        .values(status='active')
    )

def downgrade():
    pass  # 数据迁移通常不可逆
```

## 6. 生产环境最佳实践

### 6.1 迁移脚本检查

```bash
# 检查是否有待执行的迁移
alembic check

# 生成 SQL（不执行）
alembic upgrade head --sql
```

### 6.2 离线迁移

```python
# env.py
def run_migrations_offline():
    """生成 SQL 脚本，不连接数据库"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()
```

```bash
# 生成 SQL 文件
alembic upgrade head --sql > migration.sql
```

### 6.3 事务控制

```python
def upgrade():
    # 大表操作，分批处理
    connection = op.get_bind()

    # 禁用自动事务
    with connection.begin():
        # 批量更新
        offset = 0
        batch_size = 1000
        while True:
            result = connection.execute(
                text(f"UPDATE users SET status = 'active' "
                     f"WHERE id IN (SELECT id FROM users WHERE status IS NULL LIMIT {batch_size})")
            )
            if result.rowcount == 0:
                break
```

### 6.4 备份策略

```bash
# 迁移前备份
pg_dump mydb > backup_$(date +%Y%m%d).sql

# 执行迁移
alembic upgrade head

# 验证
python -c "from myapp.db import engine; print(engine.execute('SELECT COUNT(*) FROM users').scalar())"
```

## 7. 常见问题

### 7.1 自动检测未发现变化

确保在 `env.py` 中正确导入了所有模型：

```python
# env.py
from myapp.db.models import Base  # 确保所有模型都被导入
from myapp.db import models  # 或导入模块触发所有模型注册

target_metadata = Base.metadata
```

### 7.2 冲突解决

```bash
# 多人开发时可能出现分支
alembic history --verbose

# 合并分支
alembic merge -m "merge heads" rev1 rev2
```

### 7.3 重置迁移历史

```bash
# 删除所有迁移文件
rm -rf alembic/versions/*

# 重新生成
alembic revision --autogenerate -m "initial"
```

## 8. 迁移脚本模板

```python
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}

def upgrade() -> None:
    ${upgrades if upgrades else "pass"}

def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```


