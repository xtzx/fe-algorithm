"""
商品路由测试
"""


def test_list_items(client):
    """获取商品列表（公开端点）"""
    response = client.get("/api/v1/items/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_items_with_pagination(client):
    """分页获取商品"""
    response = client.get("/api/v1/items/?skip=0&limit=5")
    assert response.status_code == 200
    items = response.json()
    assert len(items) <= 5


def test_list_items_with_search(client):
    """搜索商品"""
    response = client.get("/api/v1/items/?q=iPhone")
    assert response.status_code == 200


def test_list_items_with_price_filter(client):
    """价格过滤"""
    response = client.get("/api/v1/items/?min_price=100&max_price=10000")
    assert response.status_code == 200


def test_get_item_by_id(client):
    """获取单个商品"""
    response = client.get("/api/v1/items/1")
    assert response.status_code == 200
    item = response.json()
    assert "id" in item
    assert "name" in item
    assert "price" in item


def test_get_item_not_found(client):
    """商品不存在"""
    response = client.get("/api/v1/items/99999")
    assert response.status_code == 404


def test_create_item_unauthorized(client):
    """未认证时无法创建商品"""
    response = client.post(
        "/api/v1/items/",
        json={"name": "Test Item", "price": 100.0},
    )
    assert response.status_code == 401


def test_create_item(authenticated_client):
    """创建商品"""
    response = authenticated_client.post(
        "/api/v1/items/",
        json={
            "name": "Test Item",
            "description": "A test item",
            "price": "99.99",
            "quantity": 10,
        },
    )
    assert response.status_code == 201
    item = response.json()
    assert item["name"] == "Test Item"
    assert "id" in item


def test_create_item_validation_error(authenticated_client):
    """创建商品验证失败"""
    response = authenticated_client.post(
        "/api/v1/items/",
        json={"name": "", "price": -10},
    )
    assert response.status_code == 422


def test_update_item_unauthorized(client):
    """未认证时无法更新商品"""
    response = client.put(
        "/api/v1/items/1",
        json={"name": "Updated"},
    )
    assert response.status_code == 401


def test_update_item_forbidden(authenticated_client):
    """不能更新他人的商品"""
    response = authenticated_client.put(
        "/api/v1/items/1",
        json={"name": "Hacked"},
    )
    assert response.status_code == 403


def test_admin_can_update_any_item(admin_client):
    """管理员可以更新任何商品"""
    response = admin_client.put(
        "/api/v1/items/1",
        json={"name": "Admin Updated"},
    )
    assert response.status_code == 200


def test_delete_item_unauthorized(client):
    """未认证时无法删除商品"""
    response = client.delete("/api/v1/items/1")
    assert response.status_code == 401

