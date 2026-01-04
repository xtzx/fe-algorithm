"""
Redis 缓存客户端
"""

import json
from functools import lru_cache
from typing import Any, Optional

from bookmark_api.config import get_settings

settings = get_settings()


class CacheClient:
    """缓存客户端"""

    def __init__(self):
        self._client = None
        self._available = False

    def _get_client(self):
        """获取 Redis 客户端（懒加载）"""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                )
                self._client.ping()
                self._available = True
            except Exception:
                self._available = False
                self._client = None
        return self._client

    @property
    def is_available(self) -> bool:
        """检查 Redis 是否可用"""
        self._get_client()
        return self._available

    def get(self, key: str) -> Optional[str]:
        """获取值"""
        client = self._get_client()
        if not client:
            return None
        try:
            return client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """设置值"""
        client = self._get_client()
        if not client:
            return False
        try:
            ttl = ttl or settings.cache_ttl
            return bool(client.setex(key, ttl, value))
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """删除键"""
        client = self._get_client()
        if not client:
            return False
        try:
            return bool(client.delete(key))
        except Exception:
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """获取 JSON 值"""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置 JSON 值"""
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return self.set(key, json_str, ttl)
        except (TypeError, ValueError):
            return False

    def delete_pattern(self, pattern: str) -> int:
        """删除匹配模式的键"""
        client = self._get_client()
        if not client:
            return 0
        try:
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception:
            return 0

    # ==================== 书签缓存方法 ====================

    def get_bookmark(self, user_id: int, bookmark_id: int) -> Optional[dict]:
        """获取缓存的书签"""
        key = f"bookmark:{user_id}:{bookmark_id}"
        return self.get_json(key)

    def set_bookmark(self, user_id: int, bookmark_id: int, data: dict) -> bool:
        """缓存书签"""
        key = f"bookmark:{user_id}:{bookmark_id}"
        return self.set_json(key, data, ttl=300)

    def invalidate_bookmark(self, user_id: int, bookmark_id: int) -> bool:
        """使书签缓存失效"""
        key = f"bookmark:{user_id}:{bookmark_id}"
        return self.delete(key)

    def invalidate_user_bookmarks(self, user_id: int) -> int:
        """使用户所有书签缓存失效"""
        pattern = f"bookmark:{user_id}:*"
        return self.delete_pattern(pattern)


@lru_cache()
def get_cache_client() -> CacheClient:
    """获取缓存客户端（单例）"""
    return CacheClient()

