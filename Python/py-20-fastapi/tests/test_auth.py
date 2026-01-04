"""
认证路由测试
"""


def test_login_success(client):
    """登录成功"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    """密码错误"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert response.status_code == 401


def test_login_user_not_found(client):
    """用户不存在"""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "nonexistent", "password": "password"},
    )
    assert response.status_code == 401


def test_get_current_user(authenticated_client, test_user):
    """获取当前用户"""
    response = authenticated_client.get("/api/v1/auth/me")
    assert response.status_code == 200
    user = response.json()
    assert user["username"] == test_user.username


def test_get_current_user_unauthorized(client):
    """未认证时无法获取当前用户"""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_register_user(client):
    """用户注册"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123",
        },
    )
    assert response.status_code == 201
    user = response.json()
    assert user["username"] == "newuser"
    assert "password" not in user  # 不应返回密码


def test_register_duplicate_username(client):
    """重复用户名"""
    # 先注册
    client.post(
        "/api/v1/auth/register",
        json={
            "username": "duplicate",
            "email": "dup1@example.com",
            "password": "password123",
        },
    )
    # 再次注册相同用户名
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "duplicate",
            "email": "dup2@example.com",
            "password": "password123",
        },
    )
    assert response.status_code == 400


def test_register_validation_error(client):
    """注册验证失败"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "ab",  # 太短
            "email": "invalid-email",  # 无效邮箱
            "password": "123",  # 太短
        },
    )
    assert response.status_code == 422


def test_full_auth_flow(client):
    """完整的认证流程"""
    # 1. 注册
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "flowtest",
            "email": "flowtest@example.com",
            "password": "password123",
        },
    )
    assert register_response.status_code == 201

    # 2. 登录
    login_response = client.post(
        "/api/v1/auth/token",
        data={"username": "flowtest", "password": "password123"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    # 3. 访问保护资源
    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "flowtest"

