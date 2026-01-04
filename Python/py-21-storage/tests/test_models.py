"""
模型测试
"""

from decimal import Decimal

import pytest

from storage_lab.db.models import Item, Tag, User


class TestUserModel:
    """用户模型测试"""

    def test_create_user(self, session):
        """测试创建用户"""
        user = User(
            username="john",
            email="john@example.com",
            hashed_password="hash123",
        )
        session.add(user)
        session.commit()

        assert user.id is not None
        assert user.username == "john"
        assert user.is_active is True

    def test_user_unique_username(self, session, sample_user):
        """测试用户名唯一"""
        user2 = User(
            username=sample_user.username,  # 相同用户名
            email="another@example.com",
            hashed_password="hash",
        )
        session.add(user2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_user_repr(self, sample_user):
        """测试用户表示"""
        assert "testuser" in repr(sample_user)


class TestItemModel:
    """商品模型测试"""

    def test_create_item(self, session, sample_user):
        """测试创建商品"""
        item = Item(
            name="Test Item",
            price=Decimal("99.99"),
            owner_id=sample_user.id,
        )
        session.add(item)
        session.commit()

        assert item.id is not None
        assert item.price == Decimal("99.99")
        assert item.is_available is True

    def test_item_owner_relationship(self, session, sample_user, sample_items):
        """测试商品与用户关系"""
        session.refresh(sample_user)
        assert len(sample_user.items) == 3
        assert sample_items[0].owner == sample_user


class TestTagModel:
    """标签模型测试"""

    def test_create_tag(self, session):
        """测试创建标签"""
        tag = Tag(name="python")
        session.add(tag)
        session.commit()

        assert tag.id is not None
        assert tag.name == "python"

    def test_item_tags_relationship(self, session, sample_items, sample_tags):
        """测试商品与标签多对多关系"""
        item = sample_items[0]
        item.tags.append(sample_tags[0])
        item.tags.append(sample_tags[1])
        session.commit()

        session.refresh(item)
        assert len(item.tags) == 2
        assert sample_tags[0] in item.tags

        # 反向关系
        session.refresh(sample_tags[0])
        assert item in sample_tags[0].items


