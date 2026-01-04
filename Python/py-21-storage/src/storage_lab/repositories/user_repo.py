"""
用户 Repository

提供用户相关的数据库操作
"""

from typing import List, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, joinedload

from storage_lab.db.models import User
from storage_lab.repositories.base import AsyncBaseRepository, BaseRepository


class UserRepository(BaseRepository[User]):
    """
    用户 Repository（同步）

    继承 BaseRepository，添加用户特定的查询方法
    """

    def __init__(self, session: Session):
        super().__init__(User, session)

    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return (
            self.session.query(User)
            .filter(User.email == email)
            .first()
        )

    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return (
            self.session.query(User)
            .filter(User.username == username)
            .first()
        )

    def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[User]:
        """获取所有激活用户"""
        return (
            self.session.query(User)
            .filter(User.is_active == True)  # noqa: E712
            .offset(skip)
            .limit(limit)
            .all()
        )

    def search(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[User]:
        """搜索用户（用户名或邮箱）"""
        search_pattern = f"%{query}%"
        return (
            self.session.query(User)
            .filter(
                or_(
                    User.username.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_with_items(self, user_id: int) -> Optional[User]:
        """获取用户及其商品（解决 N+1 问题）"""
        return (
            self.session.query(User)
            .options(joinedload(User.items))
            .filter(User.id == user_id)
            .first()
        )

    def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
    ) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def update_password(self, user_id: int, new_hashed_password: str) -> bool:
        """更新用户密码"""
        user = self.get_by_id(user_id)
        if user:
            user.hashed_password = new_hashed_password
            self.session.commit()
            return True
        return False

    def deactivate(self, user_id: int) -> bool:
        """停用用户"""
        user = self.get_by_id(user_id)
        if user:
            user.is_active = False
            self.session.commit()
            return True
        return False


class AsyncUserRepository(AsyncBaseRepository[User]):
    """
    用户 Repository（异步）
    """

    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_with_items(self, user_id: int) -> Optional[User]:
        """获取用户及其商品（预加载）"""
        result = await self.session.execute(
            select(User)
            .options(joinedload(User.items))
            .where(User.id == user_id)
        )
        return result.unique().scalar_one_or_none()

    async def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
    ) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user


