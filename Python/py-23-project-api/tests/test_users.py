"""
用户测试
"""

import pytest


class TestUserProfile:
    """用户资料测试"""

    def test_get_current_user(self, client, auth_headers, test_user):
        """测试获取当前用户"""
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

    def test_get_current_user_unauthorized(self, client):
        """测试未认证获取用户"""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401

    def test_update_current_user(self, client, auth_headers):
        """测试更新用户信息"""
        response = client.put(
            "/api/v1/users/me",
            json={"username": "updateduser"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["username"] == "updateduser"

    def test_update_user_duplicate_username(self, client, db, auth_headers):
        """测试更新为已存在的用户名"""
        from bookmark_api.auth.password import get_password_hash
        from bookmark_api.db.models import User

        # 创建另一个用户
        other_user = User(
            username="otheruser",
            email="other@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db.add(other_user)
        db.commit()

        # 尝试更新为已存在的用户名
        response = client.put(
            "/api/v1/users/me",
            json={"username": "otheruser"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "already taken" in response.json()["detail"]

    def test_delete_current_user(self, client, auth_headers):
        """测试删除（停用）用户"""
        response = client.delete("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200

        # 尝试再次访问
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 403  # 用户已停用

