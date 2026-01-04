"""
配置管理模块

使用 pydantic-settings 实现配置管理：
- 支持环境变量
- 支持 .env 文件
- 类型验证和转换
- 嵌套配置
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置

    配置优先级（从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值

    使用示例：
        settings = get_settings()
        print(settings.app_name)
        print(settings.debug)
    """

    # 模型配置
    model_config = SettingsConfigDict(
        # 环境变量前缀（可选）
        # env_prefix="SCAFFOLD_",
        # .env 文件路径
        env_file=".env",
        # .env 文件编码
        env_file_encoding="utf-8",
        # 嵌套模型的分隔符
        env_nested_delimiter="__",
        # 大小写敏感
        case_sensitive=False,
        # 额外字段处理
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # 应用基础配置
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="scaffold",
        description="应用名称",
    )

    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="应用环境",
    )

    debug: bool = Field(
        default=False,
        description="调试模式",
    )

    # -------------------------------------------------------------------------
    # 日志配置
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别",
    )

    log_format: Literal["text", "json"] = Field(
        default="text",
        description="日志格式",
    )

    # -------------------------------------------------------------------------
    # 数据库配置（示例）
    # -------------------------------------------------------------------------
    database_url: str | None = Field(
        default=None,
        description="数据库连接 URL",
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="数据库连接池大小",
    )

    # -------------------------------------------------------------------------
    # API 配置（示例）
    # -------------------------------------------------------------------------
    api_host: str = Field(
        default="0.0.0.0",
        description="API 监听地址",
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API 监听端口",
    )

    # -------------------------------------------------------------------------
    # 密钥配置（示例）
    # -------------------------------------------------------------------------
    secret_key: str | None = Field(
        default=None,
        description="应用密钥",
    )

    # -------------------------------------------------------------------------
    # 计算属性
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.app_env == "development"

    # -------------------------------------------------------------------------
    # 验证器
    # -------------------------------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        """确保日志级别为大写"""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("secret_key", mode="after")
    @classmethod
    def validate_secret_key_in_production(cls, v: str | None) -> str | None:
        """生产环境必须设置密钥（此处简化处理）"""
        # 实际项目中可以根据需要启用此检查
        # if settings.is_production and not v:
        #     raise ValueError("SECRET_KEY is required in production")
        return v


class DatabaseSettings(BaseSettings):
    """
    数据库配置（嵌套配置示例）

    使用方式：
        DATABASE__URL=postgresql://...
        DATABASE__POOL_SIZE=10
    """

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
    )

    url: str = Field(
        default="sqlite:///./app.db",
        description="数据库 URL",
    )

    pool_size: int = Field(
        default=5,
        description="连接池大小",
    )

    echo: bool = Field(
        default=False,
        description="是否打印 SQL",
    )


@lru_cache
def get_settings() -> Settings:
    """
    获取配置单例

    使用 lru_cache 确保配置只加载一次

    Returns:
        Settings 实例
    """
    return Settings()


def get_database_settings() -> DatabaseSettings:
    """获取数据库配置"""
    return DatabaseSettings()


# 便捷访问
settings = get_settings()


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 显示当前配置
    s = get_settings()

    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"应用名称: {s.app_name}")
    print(f"应用环境: {s.app_env}")
    print(f"调试模式: {s.debug}")
    print(f"日志级别: {s.log_level}")
    print(f"日志格式: {s.log_format}")
    print(f"是否生产: {s.is_production}")
    print(f"API 地址: {s.api_host}:{s.api_port}")

    # 显示所有配置（JSON 格式）
    print("\n完整配置 (JSON):")
    print(s.model_dump_json(indent=2))

