"""
缓存装饰器

提供:
- @cached - 简单缓存
- @cache_aside - Cache-Aside 模式
"""

import functools
import hashlib
import json
from typing import Any, Callable, Optional

from storage_lab.cache.client import CacheClient, get_cache_client


def _make_cache_key(prefix: str, func: Callable, args: tuple, kwargs: dict) -> str:
    """生成缓存键"""
    # 将参数序列化为字符串
    key_parts = [prefix, func.__module__, func.__name__]
    
    # 添加位置参数
    for arg in args:
        key_parts.append(str(arg))
    
    # 添加关键字参数（排序以保证一致性）
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    # 生成哈希
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    ttl: int = 300,
    prefix: str = "cache",
    cache_client: Optional[CacheClient] = None,
):
    """
    简单缓存装饰器
    
    Usage:
        @cached(ttl=60)
        def get_user(user_id: int):
            return db.get_user(user_id)
    
    Args:
        ttl: 缓存时间（秒）
        prefix: 缓存键前缀
        cache_client: 缓存客户端（可选）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = cache_client or get_cache_client()
            
            # 生成缓存键
            cache_key = _make_cache_key(prefix, func, args, kwargs)
            
            # 尝试从缓存获取
            cached_value = client.get_json(cache_key)
            if cached_value is not None:
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            if result is not None:
                client.set_json(cache_key, result, ttl)
            
            return result
        
        # 添加清除缓存的方法
        def invalidate(*args, **kwargs):
            client = cache_client or get_cache_client()
            cache_key = _make_cache_key(prefix, func, args, kwargs)
            client.delete(cache_key)
        
        wrapper.invalidate = invalidate
        return wrapper
    
    return decorator


def cache_aside(
    get_key: Callable[..., str],
    ttl: int = 300,
    cache_client: Optional[CacheClient] = None,
):
    """
    Cache-Aside 模式装饰器
    
    手动指定缓存键的生成方式
    
    Usage:
        @cache_aside(get_key=lambda user_id: f"user:{user_id}", ttl=60)
        def get_user(user_id: int):
            return db.get_user(user_id)
    
    Args:
        get_key: 生成缓存键的函数
        ttl: 缓存时间（秒）
        cache_client: 缓存客户端（可选）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = cache_client or get_cache_client()
            
            # 生成缓存键
            cache_key = get_key(*args, **kwargs)
            
            # 尝试从缓存获取
            cached_value = client.get_json(cache_key)
            if cached_value is not None:
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            if result is not None:
                client.set_json(cache_key, result, ttl)
            
            return result
        
        # 添加手动操作方法
        def set_cache(*args, value: Any, **kwargs):
            """手动设置缓存"""
            client = cache_client or get_cache_client()
            cache_key = get_key(*args, **kwargs)
            client.set_json(cache_key, value, ttl)
        
        def invalidate(*args, **kwargs):
            """清除缓存"""
            client = cache_client or get_cache_client()
            cache_key = get_key(*args, **kwargs)
            client.delete(cache_key)
        
        def get_cached(*args, **kwargs) -> Optional[Any]:
            """只从缓存获取，不执行函数"""
            client = cache_client or get_cache_client()
            cache_key = get_key(*args, **kwargs)
            return client.get_json(cache_key)
        
        wrapper.set_cache = set_cache
        wrapper.invalidate = invalidate
        wrapper.get_cached = get_cached
        
        return wrapper
    
    return decorator


class CacheManager:
    """
    缓存管理器
    
    提供更灵活的缓存操作
    
    Usage:
        manager = CacheManager(prefix="myapp")
        
        # 获取或设置
        user = manager.get_or_set(
            key="user:1",
            fetch_func=lambda: db.get_user(1),
            ttl=60,
        )
        
        # 清除
        manager.delete("user:1")
    """
    
    def __init__(
        self,
        prefix: str = "app",
        cache_client: Optional[CacheClient] = None,
        default_ttl: int = 300,
    ):
        self.prefix = prefix
        self.client = cache_client or get_cache_client()
        self.default_ttl = default_ttl
    
    def _make_key(self, key: str) -> str:
        """生成完整的缓存键"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        return self.client.get_json(self._make_key(key))
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        return self.client.set_json(
            self._make_key(key),
            value,
            ttl or self.default_ttl,
        )
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        return self.client.delete(self._make_key(key))
    
    def get_or_set(
        self,
        key: str,
        fetch_func: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """获取或设置（Cache-Aside 模式）"""
        # 尝试从缓存获取
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # 执行获取函数
        value = fetch_func()
        
        # 存入缓存
        if value is not None:
            self.set(key, value, ttl)
        
        return value
    
    def invalidate_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        full_pattern = f"{self.prefix}:{pattern}"
        return self.client.delete_pattern(full_pattern)


