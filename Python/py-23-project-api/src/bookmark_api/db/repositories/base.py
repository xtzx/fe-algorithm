"""
基础 Repository
"""

from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bookmark_api.db.models import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """通用 Repository 基类"""

    def __init__(self, model: Type[ModelType], session: Session):
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

