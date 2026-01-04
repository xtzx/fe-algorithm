"""
配置管理

使用 pydantic-settings 管理配置
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 数据库配置
    database_url: str = "sqlite:///./storage.db"
    database_echo: bool = False  # 是否打印 SQL
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # 异步数据库配置
    async_database_url: str = "sqlite+aiosqlite:///./storage.db"

    # Redis 配置
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300  # 默认缓存 TTL（秒）

    # 任务队列配置
    queue_redis_url: str = "redis://localhost:6379/1"


@lru_cache()
def get_settings() -> Settings:
    """获取配置（缓存）"""
    return Settings()


