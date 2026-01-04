"""主模块测试"""

import pytest
from src.main import User, UserService


class TestUser:
    """User 测试"""

    def test_create_user(self):
        """测试创建用户"""
        user = User("Alice", "alice@example.com", 30)
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        assert user.age == 30

    def test_greet(self):
        """测试问候"""
        user = User("Bob", "bob@example.com")
        assert user.greet() == "Hello, Bob!"


class TestUserService:
    """UserService 测试"""

    def test_add_user(self):
        """测试添加用户"""
        service = UserService()
        user = User("Alice", "alice@example.com")
        service.add_user(user)
        assert len(service.users) == 1

    def test_add_empty_name(self):
        """测试添加空名用户"""
        service = UserService()
        user = User("", "test@example.com")
        with pytest.raises(ValueError):
            service.add_user(user)

    def test_find_user(self):
        """测试查找用户"""
        service = UserService()
        service.add_user(User("Alice", "alice@example.com"))

        found = service.find_user("Alice")
        assert found is not None
        assert found.name == "Alice"

    def test_find_nonexistent(self):
        """测试查找不存在的用户"""
        service = UserService()
        found = service.find_user("Nobody")
        assert found is None

    def test_list_users(self):
        """测试列出用户"""
        service = UserService()
        service.add_user(User("Alice", "a@example.com"))
        service.add_user(User("Bob", "b@example.com"))

        users = service.list_users()
        assert len(users) == 2

