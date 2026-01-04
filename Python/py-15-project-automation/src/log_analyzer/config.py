"""
配置管理
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_prefix="LOG_ANALYZER_",
        env_file=".env",
    )

    # 默认日志格式
    default_format: Literal["nginx", "app", "json", "auto"] = "auto"

    # 归档设置
    archive_dir: Path = Field(default=Path("./archive"))
    compress_level: int = Field(default=6, ge=1, le=9)

    # 清理设置
    default_age_days: int = Field(default=30, ge=1)

    # 状态文件
    state_file: Path = Field(default=Path(".log-analyzer-state.json"))

    # 输出设置
    use_colors: bool = True
    verbose: bool = False


def get_settings() -> Settings:
    return Settings()

