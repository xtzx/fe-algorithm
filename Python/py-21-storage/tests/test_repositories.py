"""
Repository 测试
"""

from decimal import Decimal

import pytest

from storage_lab.db.models import Item, User
from storage_lab.repositories import ItemRepository, UserRepository


class TestUserRepository:
    """用户 Repository 测试"""

    def test_get_by_id(self, session, sample_user):
        """测试根据 ID 获取"""
        repo = UserRepository(session)
        user = repo.get_by_id(sample_user.id)
        assert user is not None
        assert user.username == sample_user.username

    def test_get_by_id_not_found(self, session):
        """测试获取不存在的用户"""
        repo = UserRepository(session)
        user = repo.get_by_id(9999)
        assert user is None

    def test_get_by_email(self, session, sample_user):
        """测试根据邮箱获取"""
        repo = UserRepository(session)
        user = repo.get_by_email(sample_user.email)
        assert user is not None
        assert user.id == sample_user.id

    def test_get_by_username(self, session, sample_user):
        """测试根据用户名获取"""
        repo = UserRepository(session)
        user = repo.get_by_username(sample_user.username)
        assert user is not None
        assert user.id == sample_user.id

    def test_create_user(self, session):
        """测试创建用户"""
        repo = UserRepository(session)
        user = repo.create_user(
            username="newuser",
            email="newuser@example.com",
            hashed_password="hash",
            full_name="New User",
        )
        assert user.id is not None
        assert user.username == "newuser"

    def test_get_active_users(self, session, sample_user):
        """测试获取激活用户"""
        # 创建一个禁用用户
        inactive = User(
            username="inactive",
            email="inactive@example.com",
            hashed_password="hash",
            is_active=False,
        )
        session.add(inactive)
        session.commit()

        repo = UserRepository(session)
        active_users = repo.get_active_users()
        assert len(active_users) == 1
        assert active_users[0].id == sample_user.id

    def test_get_with_items(self, session, sample_user, sample_items):
        """测试获取用户及其商品"""
        repo = UserRepository(session)
        user = repo.get_with_items(sample_user.id)
        assert user is not None
        assert len(user.items) == 3

    def test_search(self, session, sample_user):
        """测试搜索用户"""
        repo = UserRepository(session)

        # 搜索用户名
        results = repo.search("test")
        assert len(results) == 1

        # 搜索邮箱
        results = repo.search("example")
        assert len(results) == 1

    def test_deactivate(self, session, sample_user):
        """测试停用用户"""
        repo = UserRepository(session)
        result = repo.deactivate(sample_user.id)
        assert result is True

        session.refresh(sample_user)
        assert sample_user.is_active is False


class TestItemRepository:
    """商品 Repository 测试"""

    def test_get_by_owner(self, session, sample_user, sample_items):
        """测试获取用户的商品"""
        repo = ItemRepository(session)
        items = repo.get_by_owner(sample_user.id)
        assert len(items) == 3

    def test_get_available(self, session, sample_items):
        """测试获取上架商品"""
        # 下架一个商品
        sample_items[0].is_available = False
        session.commit()

        repo = ItemRepository(session)
        items = repo.get_available()
        assert len(items) == 2

    def test_search(self, session, sample_items):
        """测试搜索商品"""
        repo = ItemRepository(session)

        # 按名称搜索
        results = repo.search("Item 1")
        assert len(results) == 1

        # 按价格过滤
        results = repo.search("Item", min_price=Decimal("15"))
        assert len(results) == 2

    def test_create_item(self, session, sample_user):
        """测试创建商品"""
        repo = ItemRepository(session)
        item = repo.create_item(
            name="New Item",
            price=Decimal("50.00"),
            owner_id=sample_user.id,
            quantity=10,
        )
        assert item.id is not None
        assert item.name == "New Item"

    def test_add_and_remove_tag(self, session, sample_items, sample_tags):
        """测试添加和移除标签"""
        repo = ItemRepository(session)
        item = sample_items[0]

        # 添加标签
        result = repo.add_tag(item.id, "electronics")
        assert result is True

        session.refresh(item)
        assert len(item.tags) == 1

        # 移除标签
        result = repo.remove_tag(item.id, "electronics")
        assert result is True

        session.refresh(item)
        assert len(item.tags) == 0

    def test_update_quantity(self, session, sample_items):
        """测试更新库存"""
        repo = ItemRepository(session)
        item = sample_items[0]
        original_quantity = item.quantity

        # 增加库存
        result = repo.update_quantity(item.id, 10)
        assert result is True

        session.refresh(item)
        assert item.quantity == original_quantity + 10

        # 减少库存
        result = repo.update_quantity(item.id, -5)
        assert result is True

        session.refresh(item)
        assert item.quantity == original_quantity + 5

    def test_update_quantity_negative_not_allowed(self, session, sample_items):
        """测试库存不能为负"""
        repo = ItemRepository(session)
        item = sample_items[0]

        # 尝试减少超过现有库存
        result = repo.update_quantity(item.id, -1000)
        assert result is False


