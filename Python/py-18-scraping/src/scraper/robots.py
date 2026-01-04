"""
robots.txt 解析模块

支持:
- 解析 robots.txt
- 检查 URL 是否允许访问
- 获取 Crawl-Delay
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx


@dataclass
class RobotRule:
    """robots.txt 规则"""

    user_agent: str
    allow: list[str] = field(default_factory=list)
    disallow: list[str] = field(default_factory=list)
    crawl_delay: float | None = None


class RobotsParser:
    """
    robots.txt 解析器

    Example:
        ```python
        parser = RobotsParser()
        parser.parse(robots_txt_content)

        if parser.is_allowed("https://example.com/page", "MyBot"):
            print("Allowed")
        ```
    """

    def __init__(self) -> None:
        self._rules: list[RobotRule] = []
        self._sitemaps: list[str] = []

    def parse(self, content: str) -> None:
        """解析 robots.txt 内容"""
        self._rules.clear()
        self._sitemaps.clear()

        current_rule: RobotRule | None = None

        for line in content.split("\n"):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue

            # 解析指令
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                # 新的 User-Agent 块
                current_rule = RobotRule(user_agent=value.lower())
                self._rules.append(current_rule)

            elif key == "allow" and current_rule:
                current_rule.allow.append(value)

            elif key == "disallow" and current_rule:
                current_rule.disallow.append(value)

            elif key == "crawl-delay" and current_rule:
                try:
                    current_rule.crawl_delay = float(value)
                except ValueError:
                    pass

            elif key == "sitemap":
                self._sitemaps.append(value)

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """
        检查 URL 是否允许访问

        Args:
            url: 要检查的 URL
            user_agent: User-Agent 名称
        """
        path = urlparse(url).path
        user_agent = user_agent.lower()

        # 查找匹配的规则
        rule = self._find_rule(user_agent)
        if rule is None:
            return True  # 没有规则则允许

        # 检查 Allow 规则（优先级更高）
        for pattern in rule.allow:
            if self._match_path(path, pattern):
                return True

        # 检查 Disallow 规则
        for pattern in rule.disallow:
            if self._match_path(path, pattern):
                return False

        return True

    def get_crawl_delay(self, user_agent: str = "*") -> float | None:
        """获取 Crawl-Delay"""
        rule = self._find_rule(user_agent.lower())
        if rule:
            return rule.crawl_delay
        return None

    @property
    def sitemaps(self) -> list[str]:
        """获取 Sitemap URL 列表"""
        return self._sitemaps

    def _find_rule(self, user_agent: str) -> RobotRule | None:
        """查找匹配的规则"""
        # 首先查找精确匹配
        for rule in self._rules:
            if rule.user_agent == user_agent:
                return rule

        # 然后查找通配符规则
        for rule in self._rules:
            if rule.user_agent == "*":
                return rule

        return None

    def _match_path(self, path: str, pattern: str) -> bool:
        """匹配路径"""
        if not pattern:
            return False

        # 简单前缀匹配
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])

        if pattern.endswith("$"):
            return path == pattern[:-1]

        return path.startswith(pattern)


class RobotsChecker:
    """
    robots.txt 检查器

    自动获取和缓存 robots.txt

    Example:
        ```python
        checker = RobotsChecker(user_agent="MyBot/1.0")

        if await checker.is_allowed("https://example.com/page"):
            print("Allowed")
        ```
    """

    def __init__(
        self,
        user_agent: str = "*",
        cache_duration: float = 3600.0,
    ) -> None:
        """
        初始化检查器

        Args:
            user_agent: User-Agent 名称
            cache_duration: 缓存时间（秒）
        """
        self.user_agent = user_agent
        self.cache_duration = cache_duration
        self._cache: dict[str, tuple[RobotsParser, float]] = {}

    async def is_allowed(self, url: str) -> bool:
        """检查 URL 是否允许访问"""
        parser = await self._get_parser(url)
        return parser.is_allowed(url, self.user_agent)

    async def get_crawl_delay(self, url: str) -> float | None:
        """获取 Crawl-Delay"""
        parser = await self._get_parser(url)
        return parser.get_crawl_delay(self.user_agent)

    async def _get_parser(self, url: str) -> RobotsParser:
        """获取或创建解析器"""
        import time

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # 检查缓存
        if base_url in self._cache:
            parser, timestamp = self._cache[base_url]
            if time.time() - timestamp < self.cache_duration:
                return parser

        # 获取 robots.txt
        robots_url = urljoin(base_url, "/robots.txt")
        parser = RobotsParser()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(robots_url)
                if response.status_code == 200:
                    parser.parse(response.text)
        except Exception:
            # 获取失败则允许所有
            pass

        self._cache[base_url] = (parser, time.time())
        return parser

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()


def check_robots_txt(
    robots_content: str,
    url: str,
    user_agent: str = "*",
) -> bool:
    """
    检查 URL 是否被 robots.txt 允许（纯函数）

    Example:
        ```python
        robots_txt = '''
        User-agent: *
        Disallow: /admin/
        '''

        allowed = check_robots_txt(robots_txt, "/page")  # True
        allowed = check_robots_txt(robots_txt, "/admin/")  # False
        ```
    """
    parser = RobotsParser()
    parser.parse(robots_content)
    return parser.is_allowed(url, user_agent)

