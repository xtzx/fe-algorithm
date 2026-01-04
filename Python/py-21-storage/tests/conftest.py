"""
测试配置和 Fixtures
"""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage_lab.db.models import Base, Item, Tag, User


@pytest.fixture
def engine():
    """创建测试引擎（内存数据库）"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine):
    """创建测试会话"""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_user(session):
    """创建示例用户"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
    )
    session.add(user)
    session.commit()
    return user


@pytest.fixture
def sample_items(session, sample_user):
    """创建示例商品"""
    items = [
        Item(
            name="Item 1",
            price=Decimal("10.00"),
            quantity=100,
            owner_id=sample_user.id,
        ),
        Item(
            name="Item 2",
            price=Decimal("20.00"),
            quantity=50,
            owner_id=sample_user.id,
        ),
        Item(
            name="Item 3",
            price=Decimal("30.00"),
            quantity=25,
            owner_id=sample_user.id,
        ),
    ]
    session.add_all(items)
    session.commit()
    return items


@pytest.fixture
def sample_tags(session):
    """创建示例标签"""
    tags = [
        Tag(name="electronics"),
        Tag(name="sale"),
        Tag(name="new"),
    ]
    session.add_all(tags)
    session.commit()
    return tags


# Redis Mock Fixture
@pytest.fixture
def fake_redis():
    """创建 Fake Redis 客户端"""
    try:
        import fakeredis
        return fakeredis.FakeStrictRedis(decode_responses=True)
    except ImportError:
        pytest.skip("fakeredis not installed")


