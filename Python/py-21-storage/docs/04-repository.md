# Repository 模式

## 概述

Repository 模式是一种数据访问抽象，提供：

1. **CRUD 统一接口** - 标准化数据操作
2. **业务逻辑分离** - 数据访问与业务解耦
3. **易于测试** - 可以 Mock 数据层
4. **依赖注入** - 便于替换实现

## 1. 基础 Repository

### 1.1 泛型基类

```python
from typing import Generic, TypeVar, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, func

ModelType = TypeVar("ModelType")

class BaseRepository(Generic[ModelType]):
    """通用 Repository 基类"""

    def __init__(self, model: type[ModelType], session: Session):
        self.model = model
        self.session = session

    def get_by_id(self, id: int) -> Optional[ModelType]:
        """根据 ID 获取"""
        return self.session.get(self.model, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """获取所有（分页）"""
        return self.session.query(self.model).offset(skip).limit(limit).all()

    def count(self) -> int:
        """计数"""
        return self.session.query(func.count(self.model.id)).scalar() or 0

    def create(self, obj_in: dict[str, Any]) -> ModelType:
        """创建"""
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        self.session.commit()
        self.session.refresh(db_obj)
        return db_obj

    def update(self, db_obj: ModelType, obj_in: dict[str, Any]) -> ModelType:
        """更新"""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        self.session.commit()
        self.session.refresh(db_obj)
        return db_obj

    def delete(self, id: int) -> bool:
        """删除"""
        obj = self.get_by_id(id)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False
```

### 1.2 具体 Repository

```python
from typing import Optional, List
from sqlalchemy.orm import joinedload

class UserRepository(BaseRepository[User]):
    """用户 Repository"""

    def __init__(self, session: Session):
        super().__init__(User, session)

    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取"""
        return self.session.query(User).filter(User.email == email).first()

    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取"""
        return self.session.query(User).filter(User.username == username).first()

    def get_active_users(self) -> List[User]:
        """获取激活用户"""
        return self.session.query(User).filter(User.is_active == True).all()

    def get_with_items(self, user_id: int) -> Optional[User]:
        """获取用户及其商品（解决 N+1）"""
        return (
            self.session.query(User)
            .options(joinedload(User.items))
            .filter(User.id == user_id)
            .first()
        )

    def search(self, query: str) -> List[User]:
        """搜索用户"""
        pattern = f"%{query}%"
        return (
            self.session.query(User)
            .filter(
                or_(
                    User.username.ilike(pattern),
                    User.email.ilike(pattern),
                )
            )
            .all()
        )
```

## 2. 异步 Repository

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

class AsyncBaseRepository(Generic[ModelType]):
    """异步 Repository 基类"""

    def __init__(self, model: type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get_by_id(self, id: int) -> Optional[ModelType]:
        return await self.session.get(self.model, id)

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, obj_in: dict[str, Any]) -> ModelType:
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update(self, db_obj: ModelType, obj_in: dict[str, Any]) -> ModelType:
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def delete(self, id: int) -> bool:
        obj = await self.get_by_id(id)
        if obj:
            await self.session.delete(obj)
            await self.session.commit()
            return True
        return False
```

## 3. 依赖注入

### 3.1 与 FastAPI 集成

```python
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_repo(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

# 使用
@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repo),
):
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404)
    return user
```

### 3.2 工厂模式

```python
class RepositoryFactory:
    """Repository 工厂"""

    def __init__(self, session: Session):
        self.session = session
        self._user_repo: Optional[UserRepository] = None
        self._item_repo: Optional[ItemRepository] = None

    @property
    def users(self) -> UserRepository:
        if self._user_repo is None:
            self._user_repo = UserRepository(self.session)
        return self._user_repo

    @property
    def items(self) -> ItemRepository:
        if self._item_repo is None:
            self._item_repo = ItemRepository(self.session)
        return self._item_repo

# 使用
def get_repos(db: Session = Depends(get_db)) -> RepositoryFactory:
    return RepositoryFactory(db)

@app.get("/users/{user_id}/items")
def get_user_items(
    user_id: int,
    repos: RepositoryFactory = Depends(get_repos),
):
    user = repos.users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404)
    return repos.items.get_by_owner(user_id)
```

## 4. 测试策略

### 4.1 使用测试数据库

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db():
    """测试数据库"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    Base.metadata.drop_all(engine)

def test_create_user(db):
    repo = UserRepository(db)
    user = repo.create({
        "username": "test",
        "email": "test@example.com",
        "hashed_password": "hash",
    })

    assert user.id is not None
    assert user.username == "test"
```

### 4.2 Mock Repository

```python
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_user_repo():
    repo = Mock(spec=UserRepository)
    repo.get_by_id.return_value = User(
        id=1,
        username="test",
        email="test@example.com",
    )
    return repo

def test_get_user_endpoint(mock_user_repo):
    # 覆盖依赖
    app.dependency_overrides[get_user_repo] = lambda: mock_user_repo

    response = client.get("/users/1")

    assert response.status_code == 200
    mock_user_repo.get_by_id.assert_called_once_with(1)

    # 清理
    app.dependency_overrides.clear()
```

## 5. 事务管理

### 5.1 Unit of Work 模式

```python
class UnitOfWork:
    """工作单元"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()
        self.users = UserRepository(self.session)
        self.items = ItemRepository(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

# 使用
with UnitOfWork(SessionLocal) as uow:
    user = uow.users.create({"username": "test", ...})
    item = uow.items.create({"name": "item", "owner_id": user.id, ...})
    uow.commit()
```

## 6. 最佳实践

1. **单一职责** - 每个 Repository 只负责一个实体
2. **抽象接口** - 定义接口便于替换实现
3. **避免业务逻辑** - Repository 只做数据访问
4. **批量操作** - 提供批量方法提高效率
5. **懒加载策略** - 根据需要选择加载策略


