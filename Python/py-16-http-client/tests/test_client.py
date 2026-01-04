"""HTTP 客户端测试"""

import pytest

from http_kit import HttpClient
from http_kit.testing import MockResponse, MockTransport


class TestHttpClient:
    """同步 HTTP 客户端测试"""

    def test_get_request(self) -> None:
        transport = MockTransport([
            MockResponse(url="/users", json_data=[{"id": 1, "name": "Alice"}]),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        response = client.get("/users")

        assert response.status_code == 200
        assert response.json() == [{"id": 1, "name": "Alice"}]

    def test_post_request(self) -> None:
        transport = MockTransport([
            MockResponse(
                url="/users",
                method="POST",
                status_code=201,
                json_data={"id": 2, "name": "Bob"},
            ),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        response = client.post("/users", json={"name": "Bob"})

        assert response.status_code == 201
        assert response.json()["id"] == 2

    def test_put_request(self) -> None:
        transport = MockTransport([
            MockResponse(
                url="/users/1",
                method="PUT",
                json_data={"id": 1, "name": "Updated"},
            ),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        response = client.put("/users/1", json={"name": "Updated"})

        assert response.status_code == 200

    def test_delete_request(self) -> None:
        transport = MockTransport([
            MockResponse(
                url="/users/1",
                method="DELETE",
                status_code=204,
            ),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        response = client.delete("/users/1")

        assert response.status_code == 204

    def test_request_with_params(self) -> None:
        transport = MockTransport([
            MockResponse(url="/search", json_data={"results": []}),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        client.get("/search", params={"q": "python", "page": 1})

        request = transport.requests[0]
        assert "q=python" in str(request.url)
        assert "page=1" in str(request.url)

    def test_request_with_headers(self) -> None:
        transport = MockTransport([
            MockResponse(url="/", json_data={}),
        ])

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        client.get("/", headers={"X-Custom-Header": "value"})

        request = transport.requests[0]
        assert request.headers["X-Custom-Header"] == "value"

    def test_context_manager(self) -> None:
        transport = MockTransport([
            MockResponse(url="/", json_data={}),
        ])

        with HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        ) as client:
            response = client.get("/")
            assert response.status_code == 200

    def test_not_found(self) -> None:
        transport = MockTransport([], default_status=404)

        client = HttpClient(
            base_url="https://api.example.com",
            transport=transport,
        )

        response = client.get("/nonexistent")

        assert response.status_code == 404

