"""
分布式锁

提供:
- DistributedLock 类
- distributed_lock 上下文管理器
"""

import time
import uuid
from contextlib import contextmanager
from typing import Optional

from storage_lab.cache.client import CacheClient, get_cache_client


class DistributedLock:
    """
    分布式锁
    
    使用 Redis SET NX 实现
    
    Usage:
        lock = DistributedLock(client, "my-lock", timeout=10)
        
        if lock.acquire():
            try:
                # 执行临界区代码
                ...
            finally:
                lock.release()
    """
    
    def __init__(
        self,
        client: CacheClient,
        name: str,
        timeout: int = 10,
        retry_times: int = 3,
        retry_delay: float = 0.1,
    ):
        """
        Args:
            client: 缓存客户端
            name: 锁名称
            timeout: 锁超时时间（秒）
            retry_times: 获取锁的重试次数
            retry_delay: 重试间隔（秒）
        """
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = timeout
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        
        # 生成唯一标识，防止误删其他进程的锁
        self._token = str(uuid.uuid4())
        self._locked = False
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        获取锁
        
        Args:
            blocking: 是否阻塞等待
        
        Returns:
            是否成功获取锁
        """
        for _ in range(self.retry_times if blocking else 1):
            # SET key value NX EX timeout
            result = self.client.client.set(
                self.name,
                self._token,
                nx=True,  # 只在键不存在时设置
                ex=self.timeout,
            )
            
            if result:
                self._locked = True
                return True
            
            if blocking:
                time.sleep(self.retry_delay)
        
        return False
    
    def release(self) -> bool:
        """
        释放锁
        
        使用 Lua 脚本确保原子性（只释放自己的锁）
        """
        if not self._locked:
            return False
        
        # Lua 脚本：检查 token 匹配才删除
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = self.client.client.eval(lua_script, 1, self.name, self._token)
        self._locked = False
        return bool(result)
    
    def extend(self, additional_time: int) -> bool:
        """
        延长锁的过期时间
        
        Args:
            additional_time: 额外的时间（秒）
        """
        if not self._locked:
            return False
        
        # 检查是否还是自己的锁
        current_token = self.client.get(self.name)
        if current_token != self._token:
            self._locked = False
            return False
        
        return self.client.expire(self.name, self.timeout + additional_time)
    
    @property
    def locked(self) -> bool:
        """检查锁是否被持有（任何进程）"""
        return self.client.exists(self.name)
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


@contextmanager
def distributed_lock(
    name: str,
    timeout: int = 10,
    client: Optional[CacheClient] = None,
    blocking: bool = True,
):
    """
    分布式锁上下文管理器
    
    Usage:
        with distributed_lock("my-resource", timeout=10) as acquired:
            if acquired:
                # 执行临界区代码
                ...
            else:
                # 获取锁失败
                ...
    
    Args:
        name: 锁名称
        timeout: 锁超时时间（秒）
        client: 缓存客户端（可选）
        blocking: 是否阻塞等待
    
    Yields:
        是否成功获取锁
    """
    cache_client = client or get_cache_client()
    lock = DistributedLock(cache_client, name, timeout=timeout)
    
    acquired = lock.acquire(blocking=blocking)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


class RedisLock:
    """
    使用 redis-py 内置锁
    
    redis-py 提供了更完善的锁实现
    
    Usage:
        lock = RedisLock(client, "my-lock")
        with lock:
            # 临界区
            ...
    """
    
    def __init__(
        self,
        client: CacheClient,
        name: str,
        timeout: int = 10,
        blocking_timeout: Optional[float] = None,
    ):
        self.lock = client.client.lock(
            f"lock:{name}",
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )
    
    def acquire(self, blocking: bool = True) -> bool:
        return self.lock.acquire(blocking=blocking)
    
    def release(self):
        self.lock.release()
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


