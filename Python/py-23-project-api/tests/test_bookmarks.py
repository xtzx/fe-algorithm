"""
书签测试
"""

import pytest


class TestBookmarkCRUD:
    """书签 CRUD 测试"""

    def test_create_bookmark(self, client, auth_headers):
        """测试创建书签"""
        response = client.post(
            "/api/v1/bookmarks",
            json={
                "url": "https://example.com",
                "title": "Example Website",
                "description": "An example website",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["url"] == "https://example.com/"
        assert data["title"] == "Example Website"

    def test_create_bookmark_unauthorized(self, client):
        """测试未认证创建书签"""
        response = client.post(
            "/api/v1/bookmarks",
            json={
                "url": "https://example.com",
                "title": "Example",
            },
        )
        assert response.status_code == 401

    def test_list_bookmarks(self, client, auth_headers):
        """测试获取书签列表"""
        # 先创建一些书签
        for i in range(3):
            client.post(
                "/api/v1/bookmarks",
                json={
                    "url": f"https://example{i}.com",
                    "title": f"Example {i}",
                },
                headers=auth_headers,
            )

        response = client.get("/api/v1/bookmarks", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_bookmarks_pagination(self, client, auth_headers):
        """测试分页"""
        # 创建 5 个书签
        for i in range(5):
            client.post(
                "/api/v1/bookmarks",
                json={
                    "url": f"https://example{i}.com",
                    "title": f"Example {i}",
                },
                headers=auth_headers,
            )

        # 获取第一页（2 条）
        response = client.get(
            "/api/v1/bookmarks?page=1&page_size=2",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["pages"] == 3

    def test_get_bookmark(self, client, auth_headers):
        """测试获取单个书签"""
        # 先创建
        create_response = client.post(
            "/api/v1/bookmarks",
            json={
                "url": "https://example.com",
                "title": "Example",
            },
            headers=auth_headers,
        )
        bookmark_id = create_response.json()["id"]

        # 获取
        response = client.get(
            f"/api/v1/bookmarks/{bookmark_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == bookmark_id

    def test_get_bookmark_not_found(self, client, auth_headers):
        """测试获取不存在的书签"""
        response = client.get(
            "/api/v1/bookmarks/9999",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_update_bookmark(self, client, auth_headers):
        """测试更新书签"""
        # 先创建
        create_response = client.post(
            "/api/v1/bookmarks",
            json={
                "url": "https://example.com",
                "title": "Example",
            },
            headers=auth_headers,
        )
        bookmark_id = create_response.json()["id"]

        # 更新
        response = client.put(
            f"/api/v1/bookmarks/{bookmark_id}",
            json={"title": "Updated Title"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"

    def test_delete_bookmark(self, client, auth_headers):
        """测试删除书签"""
        # 先创建
        create_response = client.post(
            "/api/v1/bookmarks",
            json={
                "url": "https://example.com",
                "title": "Example",
            },
            headers=auth_headers,
        )
        bookmark_id = create_response.json()["id"]

        # 删除
        response = client.delete(
            f"/api/v1/bookmarks/{bookmark_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200

        # 确认已删除
        get_response = client.get(
            f"/api/v1/bookmarks/{bookmark_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 404


class TestBookmarkSearch:
    """书签搜索测试"""

    def test_search_by_title(self, client, auth_headers):
        """测试按标题搜索"""
        # 创建书签
        client.post(
            "/api/v1/bookmarks",
            json={"url": "https://python.org", "title": "Python Official"},
            headers=auth_headers,
        )
        client.post(
            "/api/v1/bookmarks",
            json={"url": "https://fastapi.tiangolo.com", "title": "FastAPI Docs"},
            headers=auth_headers,
        )

        # 搜索
        response = client.get(
            "/api/v1/bookmarks/search?q=python",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "Python" in data[0]["title"]


class TestBookmarkFavorite:
    """书签收藏测试"""

    def test_toggle_favorite(self, client, auth_headers):
        """测试切换收藏状态"""
        # 创建书签
        create_response = client.post(
            "/api/v1/bookmarks",
            json={"url": "https://example.com", "title": "Example"},
            headers=auth_headers,
        )
        bookmark_id = create_response.json()["id"]
        assert create_response.json()["is_favorite"] is False

        # 切换收藏
        response = client.post(
            f"/api/v1/bookmarks/{bookmark_id}/favorite",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["is_favorite"] is True

        # 再次切换
        response = client.post(
            f"/api/v1/bookmarks/{bookmark_id}/favorite",
            headers=auth_headers,
        )
        assert response.json()["is_favorite"] is False

