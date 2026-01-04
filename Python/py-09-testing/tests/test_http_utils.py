"""HTTP 工具测试

演示 mock、异步测试、集成测试。
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock

from testing_lab.http_utils import (
    fetch_json,
    async_fetch_json,
    fetch_multiple,
    GitHubClient,
    WeatherClient,
    APIError,
)


# ============================================================
# 同步 HTTP 测试（使用 mock）
# ============================================================

class TestFetchJson:
    """fetch_json 测试"""

    @patch("testing_lab.http_utils.httpx.get")
    def test_fetch_json_success(self, mock_get):
        """测试成功获取 JSON"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_json("https://api.example.com/data")

        assert result == {"data": "test"}
        mock_get.assert_called_once_with("https://api.example.com/data", timeout=10.0)

    @patch("testing_lab.http_utils.httpx.get")
    def test_fetch_json_http_error(self, mock_get):
        """测试 HTTP 错误"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(APIError, match="HTTP 错误: 404"):
            fetch_json("https://api.example.com/notfound")

    @patch("testing_lab.http_utils.httpx.get")
    def test_fetch_json_request_error(self, mock_get):
        """测试请求错误"""
        mock_get.side_effect = httpx.RequestError("Connection failed")

        with pytest.raises(APIError, match="请求错误"):
            fetch_json("https://api.example.com/data")


# ============================================================
# 异步 HTTP 测试
# ============================================================

class TestAsyncFetch:
    """异步获取测试"""

    @pytest.mark.asyncio
    async def test_async_fetch_json_success(self):
        """测试异步获取成功"""
        with patch("testing_lab.http_utils.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = {"data": "async_test"}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response

            result = await async_fetch_json("https://api.example.com/data")

            assert result == {"data": "async_test"}

    @pytest.mark.asyncio
    async def test_fetch_multiple(self):
        """测试并发获取"""
        with patch("testing_lab.http_utils.async_fetch_json", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [
                {"url": "url1"},
                {"url": "url2"},
                {"url": "url3"},
            ]

            urls = ["url1", "url2", "url3"]
            results = await fetch_multiple(urls)

            assert len(results) == 3
            assert mock_fetch.call_count == 3


# ============================================================
# GitHubClient 测试
# ============================================================

class TestGitHubClient:
    """GitHub 客户端测试"""

    @pytest.fixture
    def client(self):
        return GitHubClient()

    @pytest.fixture
    def authenticated_client(self):
        return GitHubClient(token="test-token")

    @patch("testing_lab.http_utils.httpx.request")
    def test_get_user(self, mock_request, client):
        """测试获取用户"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "login": "octocat",
            "name": "The Octocat",
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = client.get_user("octocat")

        assert result["login"] == "octocat"
        mock_request.assert_called_once()

    @patch("testing_lab.http_utils.httpx.request")
    def test_get_repo(self, mock_request, client):
        """测试获取仓库"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "hello-world",
            "full_name": "octocat/hello-world",
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = client.get_repo("octocat", "hello-world")

        assert result["full_name"] == "octocat/hello-world"

    @patch("testing_lab.http_utils.httpx.request")
    def test_user_not_found(self, mock_request, client):
        """测试用户不存在"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response
        )
        mock_request.return_value = mock_response

        with pytest.raises(APIError, match="资源不存在"):
            client.get_user("nonexistent")

    @patch("testing_lab.http_utils.httpx.request")
    def test_authenticated_request(self, mock_request, authenticated_client):
        """测试认证请求"""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        authenticated_client.get_user("octocat")

        # 验证 headers 包含 token
        call_kwargs = mock_request.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert "test-token" in call_kwargs["headers"]["Authorization"]


# ============================================================
# WeatherClient 测试
# ============================================================

class TestWeatherClient:
    """天气客户端测试"""

    @pytest.fixture
    def client(self):
        return WeatherClient(api_key="test-key")

    @patch("testing_lab.http_utils.httpx.get")
    def test_get_current_weather(self, mock_get, client):
        """测试获取当前天气"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "temperature": 25,
            "condition": "sunny",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = client.get_current("Beijing")

        assert result["temperature"] == 25
        assert result["condition"] == "sunny"

        # 验证请求参数
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"]["city"] == "Beijing"
        assert call_kwargs["params"]["key"] == "test-key"

    @patch("testing_lab.http_utils.httpx.get")
    def test_get_forecast(self, mock_get, client):
        """测试获取天气预报"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "forecast": [
                {"day": 1, "temp": 25},
                {"day": 2, "temp": 26},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = client.get_forecast("Beijing", days=2)

        assert len(result) == 2
        assert result[0]["temp"] == 25


# ============================================================
# 标记测试
# ============================================================

@pytest.mark.integration
class TestIntegration:
    """集成测试（标记为 integration）"""

    @pytest.mark.skip(reason="需要真实 API")
    def test_real_github_api(self):
        """真实 GitHub API 测试"""
        client = GitHubClient()
        result = client.get_user("octocat")
        assert result["login"] == "octocat"


@pytest.mark.slow
class TestSlow:
    """慢速测试示例"""

    @pytest.mark.skip(reason="示例")
    def test_slow_operation(self):
        import time
        time.sleep(5)
        assert True

