"""
用户服务

提供用户相关业务逻辑:
- 用户查询
- 用户创建
- 用户更新
- 用户删除
"""

from datetime import datetime

from api.schemas.user import User, UserInDB, UserUpdate


# 模拟数据库（实际项目中应使用真实数据库）
FAKE_USERS_DB: dict[int, dict] = {
    1: {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "管理员",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "scopes": ["admin", "user"],
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "updated_at": None,
    },
    2: {
        "id": 2,
        "username": "user1",
        "email": "user1@example.com",
        "full_name": "普通用户",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "scopes": ["user"],
        "created_at": datetime(2024, 1, 2, 0, 0, 0),
        "updated_at": None,
    },
    3: {
        "id": 3,
        "username": "inactive",
        "email": "inactive@example.com",
        "full_name": "禁用用户",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": False,
        "scopes": ["user"],
        "created_at": datetime(2024, 1, 3, 0, 0, 0),
        "updated_at": None,
    },
}


class UserService:
    """用户服务类"""

    def __init__(self):
        self._next_id = max(FAKE_USERS_DB.keys()) + 1 if FAKE_USERS_DB else 1

    def get_users(self, skip: int = 0, limit: int = 10) -> list[User]:
        """获取用户列表"""
        users = list(FAKE_USERS_DB.values())[skip : skip + limit]
        return [User(**user) for user in users]

    def get_user_by_id(self, user_id: int) -> User | None:
        """根据 ID 获取用户"""
        user_data = FAKE_USERS_DB.get(user_id)
        if user_data:
            return User(**user_data)
        return None

    def get_user_by_username(self, username: str) -> User | None:
        """根据用户名获取用户"""
        for user_data in FAKE_USERS_DB.values():
            if user_data["username"] == username:
                return User(**user_data)
        return None

    def get_user_in_db(self, username: str) -> UserInDB | None:
        """获取包含密码哈希的用户（用于认证）"""
        for user_data in FAKE_USERS_DB.values():
            if user_data["username"] == username:
                return UserInDB(**user_data)
        return None

    def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: str | None = None,
    ) -> User:
        """创建用户"""
        user_data = {
            "id": self._next_id,
            "username": username,
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "is_active": True,
            "scopes": ["user"],
            "created_at": datetime.now(),
            "updated_at": None,
        }
        FAKE_USERS_DB[self._next_id] = user_data
        self._next_id += 1
        return User(**user_data)

    def update_user(self, user_id: int, user_data: UserUpdate) -> User | None:
        """更新用户"""
        if user_id not in FAKE_USERS_DB:
            return None

        update_dict = user_data.model_dump(exclude_unset=True)
        if update_dict:
            FAKE_USERS_DB[user_id].update(update_dict)
            FAKE_USERS_DB[user_id]["updated_at"] = datetime.now()

        return User(**FAKE_USERS_DB[user_id])

    def delete_user(self, user_id: int) -> bool:
        """删除用户"""
        if user_id in FAKE_USERS_DB:
            del FAKE_USERS_DB[user_id]
            return True
        return False

