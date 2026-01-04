"""
基础 Repository

提供:
- 通用 CRUD 操作
- 分页查询
- 事务管理
"""

from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from storage_lab.db.models import Base

# 类型变量
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    通用 Repository 基类

    提供 CRUD 基础操作

    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, session: Session):
                super().__init__(User, session)
    """

    def __init__(self, model: Type[ModelType], session: Session):
        self.model = model
        self.session = session

    def get_by_id(self, id: int) -> Optional[ModelType]:
        """根据 ID 获取单条记录"""
        return self.session.get(self.model, id)

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ModelType]:
        """获取所有记录（分页）"""
        return (
            self.session.query(self.model)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count(self) -> int:
        """获取记录总数"""
        return self.session.query(func.count(self.model.id)).scalar() or 0

    def create(self, obj_in: dict[str, Any]) -> ModelType:
        """创建记录"""
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        self.session.commit()
        self.session.refresh(db_obj)
        return db_obj

    def update(
        self,
        db_obj: ModelType,
        obj_in: dict[str, Any],
    ) -> ModelType:
        """更新记录"""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        self.session.commit()
        self.session.refresh(db_obj)
        return db_obj

    def delete(self, id: int) -> bool:
        """删除记录"""
        obj = self.get_by_id(id)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False

    def bulk_create(self, objs_in: List[dict[str, Any]]) -> List[ModelType]:
        """批量创建"""
        db_objs = [self.model(**obj_in) for obj_in in objs_in]
        self.session.add_all(db_objs)
        self.session.commit()
        for obj in db_objs:
            self.session.refresh(obj)
        return db_objs


class AsyncBaseRepository(Generic[ModelType]):
    """
    异步 Repository 基类

    Usage:
        class AsyncUserRepository(AsyncBaseRepository[User]):
            def __init__(self, session: AsyncSession):
                super().__init__(User, session)
    """

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get_by_id(self, id: int) -> Optional[ModelType]:
        """根据 ID 获取单条记录"""
        return await self.session.get(self.model, id)

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ModelType]:
        """获取所有记录（分页）"""
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """获取记录总数"""
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar() or 0

    async def create(self, obj_in: dict[str, Any]) -> ModelType:
        """创建记录"""
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db_obj: ModelType,
        obj_in: dict[str, Any],
    ) -> ModelType:
        """更新记录"""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def delete(self, id: int) -> bool:
        """删除记录"""
        obj = await self.get_by_id(id)
        if obj:
            await self.session.delete(obj)
            await self.session.commit()
            return True
        return False


