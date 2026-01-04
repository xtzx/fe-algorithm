"""
用户 Repository
"""

from typing import Optional

from sqlalchemy.orm import Session

from bookmark_api.db.models import User
from bookmark_api.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """用户 Repository"""

    def __init__(self, session: Session):
        super().__init__(User, session)

    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取"""
        return self.session.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取"""
        return self.session.query(User).filter(User.email == email).first()

    def get_by_username_or_email(self, username: str, email: str) -> Optional[User]:
        """根据用户名或邮箱获取"""
        return (
            self.session.query(User)
            .filter((User.username == username) | (User.email == email))
            .first()
        )

    def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
    ) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def update_password(self, user_id: int, hashed_password: str) -> bool:
        """更新密码"""
        user = self.get_by_id(user_id)
        if user:
            user.hashed_password = hashed_password
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

