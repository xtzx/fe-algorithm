"""
认证测试
"""

import pytest


def test_login_success(client):
    """测试登录成功"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    """测试密码错误"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    
    assert response.status_code == 401


def test_login_user_not_found(client):
    """测试用户不存在"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "nonexistent", "password": "password"},
    )
    
    assert response.status_code == 401


def test_register_user(client):
    """测试用户注册"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
        },
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"


def test_get_current_user(client, auth_headers):
    """测试获取当前用户"""
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin"


def test_unauthorized_access(client):
    """测试未授权访问"""
    response = client.get("/api/v1/auth/me")
    
    assert response.status_code == 401


