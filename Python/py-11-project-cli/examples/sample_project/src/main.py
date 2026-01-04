#!/usr/bin/env python3
"""示例项目主模块

这是一个用于测试代码统计工具的示例项目。
"""

import logging
from dataclasses import dataclass
from typing import Optional

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class User:
    """用户数据类"""

    name: str
    email: str
    age: int = 0

    def greet(self) -> str:
        """返回问候语"""
        return f"Hello, {self.name}!"


class UserService:
    """用户服务类"""

    def __init__(self):
        self.users: list[User] = []

    def add_user(self, user: User) -> None:
        """添加用户"""
        # 验证用户
        if not user.name:
            raise ValueError("用户名不能为空")
        self.users.append(user)
        logger.info(f"Added user: {user.name}")

    def find_user(self, name: str) -> Optional[User]:
        """查找用户"""
        for user in self.users:
            if user.name == name:
                return user
        return None

    def list_users(self) -> list[User]:
        """列出所有用户"""
        return self.users.copy()


def main() -> int:
    """主函数"""
    # 创建服务
    service = UserService()

    # 添加用户
    service.add_user(User("Alice", "alice@example.com", 30))
    service.add_user(User("Bob", "bob@example.com", 25))

    # 查找用户
    alice = service.find_user("Alice")
    if alice:
        print(alice.greet())

    # 列出所有用户
    print(f"Total users: {len(service.list_users())}")

    return 0


if __name__ == "__main__":
    exit(main())

