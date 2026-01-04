"""
测试配置和 Fixtures

提供:
- 测试客户端
- 模拟用户
- 依赖覆盖
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.dependencies.auth import get_current_active_user, get_current_user
from api.main import app
from api.schemas.user import User


@pytest.fixture
def client():
    """创建测试客户端"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_user():
    """测试用户"""
    return User(
        id=999,
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        scopes=["user"],
        created_at=datetime.now(),
        updated_at=None,
    )


@pytest.fixture
def admin_user():
    """管理员用户"""
    return User(
        id=1,
        username="admin",
        email="admin@example.com",
        full_name="Admin User",
        is_active=True,
        scopes=["user", "admin"],
        created_at=datetime.now(),
        updated_at=None,
    )


@pytest.fixture
def inactive_user():
    """禁用用户"""
    return User(
        id=998,
        username="inactive",
        email="inactive@example.com",
        full_name="Inactive User",
        is_active=False,
        scopes=["user"],
        created_at=datetime.now(),
        updated_at=None,
    )


@pytest.fixture
def authenticated_client(client, test_user):
    """已认证的测试客户端（普通用户）"""

    def override_get_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_user

    yield client

    app.dependency_overrides.clear()


@pytest.fixture
def admin_client(client, admin_user):
    """管理员测试客户端"""

    def override_get_current_user():
        return admin_user

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_user

    yield client

    app.dependency_overrides.clear()

