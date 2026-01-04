"""
配置管理

使用 pydantic-settings 管理配置
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 应用配置
    app_name: str = "Bookmark API"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000

    # 数据库配置
    database_url: str = "sqlite:///./bookmark.db"
    database_echo: bool = False

    # Redis 配置
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300

    # JWT 配置
    jwt_secret_key: str = "change-me-in-production-use-strong-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # CORS 配置
    cors_origins: List[str] = ["*"]

    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"  # json or console

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """获取配置（缓存）"""
    return Settings()

