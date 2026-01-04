"""
认证测试
"""

import pytest


class TestRegister:
    """注册测试"""

    def test_register_success(self, client):
        """测试注册成功"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"

    def test_register_duplicate_username(self, client, test_user):
        """测试重复用户名"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "another@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 400
        assert "Username already registered" in response.json()["detail"]

    def test_register_duplicate_email(self, client, test_user):
        """测试重复邮箱"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "anotheruser",
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    def test_register_invalid_email(self, client):
        """测试无效邮箱"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "invalid-email",
                "password": "password123",
            },
        )
        assert response.status_code == 422


class TestLogin:
    """登录测试"""

    def test_login_success(self, client, test_user):
        """测试登录成功"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        """测试密码错误"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_user_not_found(self, client):
        """测试用户不存在"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent", "password": "password123"},
        )
        assert response.status_code == 401


class TestRefreshToken:
    """刷新令牌测试"""

    def test_refresh_token_success(self, client, test_user):
        """测试刷新令牌成功"""
        # 先登录获取 refresh token
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "password123"},
        )
        refresh_token = login_response.json()["refresh_token"]

        # 刷新令牌
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_token_invalid(self, client):
        """测试无效刷新令牌"""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid-token"},
        )
        assert response.status_code == 401

