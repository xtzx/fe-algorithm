"""HTTP 工具模块 - 用于演示 mock 和集成测试"""

import asyncio
from typing import Any

import httpx


class APIError(Exception):
    """API 错误"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def fetch_json(url: str, timeout: float = 10.0) -> dict:
    """同步获取 JSON 数据

    Raises:
        APIError: 请求失败
    """
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"HTTP 错误: {e.response.status_code}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"请求错误: {e}")


async def async_fetch_json(url: str, timeout: float = 10.0) -> dict:
    """异步获取 JSON 数据

    Raises:
        APIError: 请求失败
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"HTTP 错误: {e.response.status_code}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"请求错误: {e}")


async def fetch_multiple(urls: list[str]) -> list[dict]:
    """并发获取多个 URL"""
    tasks = [async_fetch_json(url) for url in urls]
    return await asyncio.gather(*tasks)


class GitHubClient:
    """GitHub API 客户端"""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.token = token
        self._headers = {}
        if token:
            self._headers["Authorization"] = f"token {token}"

    def get_user(self, username: str) -> dict:
        """获取用户信息"""
        url = f"{self.BASE_URL}/users/{username}"
        return self._request("GET", url)

    def get_repo(self, owner: str, repo: str) -> dict:
        """获取仓库信息"""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        return self._request("GET", url)

    def list_repos(self, username: str) -> list[dict]:
        """列出用户的仓库"""
        url = f"{self.BASE_URL}/users/{username}/repos"
        return self._request("GET", url)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        """发送请求"""
        try:
            response = httpx.request(
                method,
                url,
                headers=self._headers,
                timeout=10.0,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise APIError("资源不存在", 404)
            elif e.response.status_code == 401:
                raise APIError("认证失败", 401)
            elif e.response.status_code == 403:
                raise APIError("权限不足或请求过于频繁", 403)
            else:
                raise APIError(f"HTTP 错误: {e.response.status_code}", e.response.status_code)
        except httpx.RequestError as e:
            raise APIError(f"请求错误: {e}")


class WeatherClient:
    """天气 API 客户端（示例）"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.weather.example.com"

    def get_current(self, city: str) -> dict:
        """获取当前天气"""
        url = f"{self.base_url}/current"
        params = {"city": city, "key": self.api_key}

        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()

    def get_forecast(self, city: str, days: int = 7) -> list[dict]:
        """获取天气预报"""
        url = f"{self.base_url}/forecast"
        params = {"city": city, "days": days, "key": self.api_key}

        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()["forecast"]

