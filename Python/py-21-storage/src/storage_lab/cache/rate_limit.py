"""
限流实现

提供:
- 固定窗口限流
- 滑动窗口限流
- 令牌桶限流
"""

import time
from typing import Optional

from storage_lab.cache.client import CacheClient, get_cache_client


class RateLimiter:
    """
    固定窗口限流器

    在固定时间窗口内限制请求次数

    Usage:
        limiter = RateLimiter(client, "api", max_requests=100, window=60)

        if limiter.is_allowed("user:123"):
            # 允许请求
            ...
        else:
            # 限流
            raise RateLimitExceeded()
    """

    def __init__(
        self,
        client: CacheClient,
        name: str,
        max_requests: int = 100,
        window: int = 60,
    ):
        """
        Args:
            client: 缓存客户端
            name: 限流器名称
            max_requests: 窗口内最大请求数
            window: 时间窗口（秒）
        """
        self.client = client
        self.name = name
        self.max_requests = max_requests
        self.window = window

    def _make_key(self, identifier: str) -> str:
        """生成限流键"""
        return f"ratelimit:{self.name}:{identifier}"

    def is_allowed(self, identifier: str) -> bool:
        """
        检查是否允许请求

        Args:
            identifier: 标识符（如用户 ID、IP 地址）

        Returns:
            是否允许请求
        """
        key = self._make_key(identifier)

        # 使用管道保证原子性
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.window)
        results = pipe.execute()

        current_count = results[0]
        return current_count <= self.max_requests

    def get_remaining(self, identifier: str) -> int:
        """获取剩余请求次数"""
        key = self._make_key(identifier)
        current = self.client.get(key)
        if current is None:
            return self.max_requests
        return max(0, self.max_requests - int(current))

    def reset(self, identifier: str) -> bool:
        """重置计数"""
        key = self._make_key(identifier)
        return self.client.delete(key)


class SlidingWindowRateLimiter:
    """
    滑动窗口限流器

    使用有序集合实现更精确的限流

    Usage:
        limiter = SlidingWindowRateLimiter(client, "api", max_requests=100, window=60)

        allowed, remaining = limiter.check_and_update("user:123")
        if allowed:
            # 处理请求
            ...
    """

    def __init__(
        self,
        client: CacheClient,
        name: str,
        max_requests: int = 100,
        window: int = 60,
    ):
        self.client = client
        self.name = name
        self.max_requests = max_requests
        self.window = window

    def _make_key(self, identifier: str) -> str:
        return f"ratelimit:sliding:{self.name}:{identifier}"

    def check_and_update(self, identifier: str) -> tuple[bool, int]:
        """
        检查并更新请求计数

        Returns:
            (是否允许, 剩余请求数)
        """
        key = self._make_key(identifier)
        now = time.time()
        window_start = now - self.window

        # 使用 Lua 脚本保证原子性
        lua_script = """
        -- 移除过期的请求
        redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])

        -- 获取当前请求数
        local count = redis.call('ZCARD', KEYS[1])

        -- 如果未超过限制，添加新请求
        if count < tonumber(ARGV[2]) then
            redis.call('ZADD', KEYS[1], ARGV[3], ARGV[3])
            redis.call('EXPIRE', KEYS[1], ARGV[4])
            return {1, tonumber(ARGV[2]) - count - 1}
        else
            return {0, 0}
        end
        """

        result = self.client.client.eval(
            lua_script,
            1,
            key,
            window_start,
            self.max_requests,
            now,
            self.window,
        )

        return bool(result[0]), int(result[1])

    def is_allowed(self, identifier: str) -> bool:
        """检查是否允许请求"""
        allowed, _ = self.check_and_update(identifier)
        return allowed

    def get_count(self, identifier: str) -> int:
        """获取当前窗口内的请求数"""
        key = self._make_key(identifier)
        now = time.time()
        window_start = now - self.window

        # 移除过期的请求并计数
        self.client.client.zremrangebyscore(key, 0, window_start)
        return self.client.client.zcard(key)


class TokenBucketRateLimiter:
    """
    令牌桶限流器

    允许突发流量，平滑限流

    Usage:
        limiter = TokenBucketRateLimiter(
            client, "api",
            capacity=100,    # 桶容量
            fill_rate=10,    # 每秒填充 10 个令牌
        )

        if limiter.consume("user:123", tokens=1):
            # 处理请求
            ...
    """

    def __init__(
        self,
        client: CacheClient,
        name: str,
        capacity: int = 100,
        fill_rate: float = 10.0,
    ):
        """
        Args:
            client: 缓存客户端
            name: 限流器名称
            capacity: 桶容量（最大令牌数）
            fill_rate: 填充速率（每秒令牌数）
        """
        self.client = client
        self.name = name
        self.capacity = capacity
        self.fill_rate = fill_rate

    def _make_key(self, identifier: str) -> str:
        return f"ratelimit:bucket:{self.name}:{identifier}"

    def consume(self, identifier: str, tokens: int = 1) -> bool:
        """
        消费令牌

        Args:
            identifier: 标识符
            tokens: 消费的令牌数

        Returns:
            是否成功消费
        """
        key = self._make_key(identifier)
        now = time.time()

        # Lua 脚本实现令牌桶算法
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local fill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_to_consume = tonumber(ARGV[4])

        -- 获取当前状态
        local bucket = redis.call('HMGET', key, 'tokens', 'last_time')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_time = tonumber(bucket[2]) or now

        -- 计算填充的令牌
        local elapsed = now - last_time
        local new_tokens = math.min(capacity, current_tokens + elapsed * fill_rate)

        -- 检查是否有足够的令牌
        if new_tokens >= tokens_to_consume then
            new_tokens = new_tokens - tokens_to_consume
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_time', now)
            redis.call('EXPIRE', key, capacity / fill_rate * 2)
            return 1
        else
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_time', now)
            redis.call('EXPIRE', key, capacity / fill_rate * 2)
            return 0
        end
        """

        result = self.client.client.eval(
            lua_script,
            1,
            key,
            self.capacity,
            self.fill_rate,
            now,
            tokens,
        )

        return bool(result)

    def get_tokens(self, identifier: str) -> float:
        """获取当前可用令牌数"""
        key = self._make_key(identifier)
        bucket = self.client.hgetall(key)

        if not bucket:
            return float(self.capacity)

        current_tokens = float(bucket.get("tokens", self.capacity))
        last_time = float(bucket.get("last_time", time.time()))

        # 计算当前令牌数
        elapsed = time.time() - last_time
        return min(self.capacity, current_tokens + elapsed * self.fill_rate)


