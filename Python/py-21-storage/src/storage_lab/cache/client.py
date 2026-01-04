"""
Redis 缓存客户端

提供:
- 基础操作
- 连接池管理
- 序列化支持
"""

import json
from functools import lru_cache
from typing import Any, Optional, Union

import redis
from redis import Redis

from storage_lab.config import get_settings


class CacheClient:
    """
    Redis 缓存客户端
    
    封装常用的缓存操作
    
    Usage:
        cache = CacheClient()
        cache.set("key", {"data": "value"}, ttl=300)
        data = cache.get("key")
    """
    
    def __init__(self, url: Optional[str] = None):
        settings = get_settings()
        self.url = url or settings.redis_url
        self.default_ttl = settings.cache_ttl
        
        # 创建连接池
        self.pool = redis.ConnectionPool.from_url(
            self.url,
            max_connections=10,
            decode_responses=True,
        )
        self._client: Optional[Redis] = None
    
    @property
    def client(self) -> Redis:
        """获取 Redis 客户端（懒加载）"""
        if self._client is None:
            self._client = Redis(connection_pool=self.pool)
        return self._client
    
    # ==================== 基础操作 ====================
    
    def get(self, key: str) -> Optional[str]:
        """获取字符串值"""
        return self.client.get(key)
    
    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置字符串值"""
        ttl = ttl or self.default_ttl
        return bool(self.client.setex(key, ttl, value))
    
    def delete(self, key: str) -> bool:
        """删除键"""
        return bool(self.client.delete(key))
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return bool(self.client.exists(key))
    
    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        return bool(self.client.expire(key, ttl))
    
    def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        return self.client.ttl(key)
    
    # ==================== JSON 操作 ====================
    
    def get_json(self, key: str) -> Optional[Any]:
        """获取 JSON 值"""
        value = self.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置 JSON 值"""
        return self.set(key, json.dumps(value, ensure_ascii=False), ttl)
    
    # ==================== 计数器 ====================
    
    def incr(self, key: str, amount: int = 1) -> int:
        """增加计数"""
        return self.client.incr(key, amount)
    
    def decr(self, key: str, amount: int = 1) -> int:
        """减少计数"""
        return self.client.decr(key, amount)
    
    # ==================== Hash 操作 ====================
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """获取 Hash 字段"""
        return self.client.hget(name, key)
    
    def hset(self, name: str, key: str, value: str) -> int:
        """设置 Hash 字段"""
        return self.client.hset(name, key, value)
    
    def hgetall(self, name: str) -> dict[str, str]:
        """获取所有 Hash 字段"""
        return self.client.hgetall(name)
    
    def hdel(self, name: str, *keys: str) -> int:
        """删除 Hash 字段"""
        return self.client.hdel(name, *keys)
    
    # ==================== List 操作 ====================
    
    def lpush(self, key: str, *values: str) -> int:
        """从左侧插入列表"""
        return self.client.lpush(key, *values)
    
    def rpush(self, key: str, *values: str) -> int:
        """从右侧插入列表"""
        return self.client.rpush(key, *values)
    
    def lpop(self, key: str) -> Optional[str]:
        """从左侧弹出"""
        return self.client.lpop(key)
    
    def rpop(self, key: str) -> Optional[str]:
        """从右侧弹出"""
        return self.client.rpop(key)
    
    def lrange(self, key: str, start: int, end: int) -> list[str]:
        """获取列表范围"""
        return self.client.lrange(key, start, end)
    
    def llen(self, key: str) -> int:
        """获取列表长度"""
        return self.client.llen(key)
    
    # ==================== Set 操作 ====================
    
    def sadd(self, key: str, *values: str) -> int:
        """添加集合元素"""
        return self.client.sadd(key, *values)
    
    def srem(self, key: str, *values: str) -> int:
        """移除集合元素"""
        return self.client.srem(key, *values)
    
    def smembers(self, key: str) -> set[str]:
        """获取所有集合元素"""
        return self.client.smembers(key)
    
    def sismember(self, key: str, value: str) -> bool:
        """检查是否是集合成员"""
        return bool(self.client.sismember(key, value))
    
    # ==================== Sorted Set 操作 ====================
    
    def zadd(
        self,
        key: str,
        mapping: dict[str, Union[int, float]],
    ) -> int:
        """添加有序集合元素"""
        return self.client.zadd(key, mapping)
    
    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
    ) -> list:
        """获取有序集合范围"""
        return self.client.zrange(key, start, end, withscores=withscores)
    
    def zrank(self, key: str, value: str) -> Optional[int]:
        """获取元素排名"""
        return self.client.zrank(key, value)
    
    # ==================== 批量操作 ====================
    
    def mget(self, *keys: str) -> list[Optional[str]]:
        """批量获取"""
        return self.client.mget(*keys)
    
    def mset(self, mapping: dict[str, str]) -> bool:
        """批量设置"""
        return bool(self.client.mset(mapping))
    
    def delete_pattern(self, pattern: str) -> int:
        """删除匹配模式的所有键"""
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0
    
    # ==================== 管道 ====================
    
    def pipeline(self):
        """获取管道（批量执行命令）"""
        return self.client.pipeline()
    
    # ==================== 连接管理 ====================
    
    def ping(self) -> bool:
        """检查连接"""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False
    
    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None


@lru_cache()
def get_cache_client() -> CacheClient:
    """获取缓存客户端（单例）"""
    return CacheClient()


