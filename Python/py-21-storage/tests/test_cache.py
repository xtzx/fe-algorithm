"""
缓存测试
"""

import json
import time

import pytest


class TestCacheClient:
    """缓存客户端测试"""

    def test_set_and_get(self, fake_redis):
        """测试设置和获取"""
        fake_redis.set("key1", "value1")
        assert fake_redis.get("key1") == "value1"

    def test_setex_with_ttl(self, fake_redis):
        """测试带过期时间的设置"""
        fake_redis.setex("key2", 10, "value2")
        assert fake_redis.get("key2") == "value2"
        assert fake_redis.ttl("key2") <= 10

    def test_delete(self, fake_redis):
        """测试删除"""
        fake_redis.set("key3", "value3")
        fake_redis.delete("key3")
        assert fake_redis.get("key3") is None

    def test_exists(self, fake_redis):
        """测试存在检查"""
        fake_redis.set("key4", "value4")
        assert fake_redis.exists("key4") == 1
        assert fake_redis.exists("nonexistent") == 0

    def test_incr_decr(self, fake_redis):
        """测试计数器"""
        fake_redis.set("counter", "0")
        assert fake_redis.incr("counter") == 1
        assert fake_redis.incr("counter", 5) == 6
        assert fake_redis.decr("counter") == 5

    def test_hash_operations(self, fake_redis):
        """测试哈希操作"""
        fake_redis.hset("user:1", "name", "John")
        fake_redis.hset("user:1", "age", "30")

        assert fake_redis.hget("user:1", "name") == "John"
        assert fake_redis.hgetall("user:1") == {"name": "John", "age": "30"}

        fake_redis.hdel("user:1", "age")
        assert fake_redis.hget("user:1", "age") is None

    def test_list_operations(self, fake_redis):
        """测试列表操作"""
        fake_redis.rpush("queue", "task1", "task2", "task3")

        assert fake_redis.llen("queue") == 3
        assert fake_redis.lpop("queue") == "task1"
        assert fake_redis.rpop("queue") == "task3"
        assert fake_redis.lrange("queue", 0, -1) == ["task2"]

    def test_set_operations(self, fake_redis):
        """测试集合操作"""
        fake_redis.sadd("tags", "python", "redis", "database")

        members = fake_redis.smembers("tags")
        assert len(members) == 3
        assert "python" in members

        assert fake_redis.sismember("tags", "python") == 1
        assert fake_redis.sismember("tags", "java") == 0

    def test_sorted_set_operations(self, fake_redis):
        """测试有序集合操作"""
        fake_redis.zadd("leaderboard", {"player1": 100, "player2": 200, "player3": 150})

        # 升序
        top = fake_redis.zrange("leaderboard", 0, -1)
        assert top == ["player1", "player3", "player2"]

        # 降序
        top_desc = fake_redis.zrevrange("leaderboard", 0, -1)
        assert top_desc == ["player2", "player3", "player1"]

        # 排名
        assert fake_redis.zrank("leaderboard", "player1") == 0

    def test_json_operations(self, fake_redis):
        """测试 JSON 存储"""
        data = {"name": "John", "age": 30, "active": True}
        fake_redis.set("user:json", json.dumps(data))

        loaded = json.loads(fake_redis.get("user:json"))
        assert loaded == data

    def test_pipeline(self, fake_redis):
        """测试管道"""
        pipe = fake_redis.pipeline()
        pipe.set("p1", "v1")
        pipe.set("p2", "v2")
        pipe.get("p1")
        pipe.get("p2")
        results = pipe.execute()

        assert results[2] == "v1"
        assert results[3] == "v2"


class TestRateLimiter:
    """限流器测试"""

    def test_fixed_window_rate_limit(self, fake_redis):
        """测试固定窗口限流"""
        # 简化的限流实现
        max_requests = 5
        window = 60

        def is_allowed(identifier: str) -> bool:
            key = f"ratelimit:{identifier}"
            pipe = fake_redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            count, _ = pipe.execute()
            return count <= max_requests

        # 前 5 次应该允许
        for _ in range(5):
            assert is_allowed("user:1") is True

        # 第 6 次应该拒绝
        assert is_allowed("user:1") is False


class TestDistributedLock:
    """分布式锁测试"""

    def test_acquire_and_release(self, fake_redis):
        """测试获取和释放锁"""
        lock_name = "test-lock"
        token = "token-123"

        # 获取锁
        result = fake_redis.set(lock_name, token, nx=True, ex=10)
        assert result is True

        # 再次获取应该失败
        result = fake_redis.set(lock_name, "another-token", nx=True, ex=10)
        assert result is None

        # 释放锁
        fake_redis.delete(lock_name)

        # 现在可以获取
        result = fake_redis.set(lock_name, token, nx=True, ex=10)
        assert result is True


