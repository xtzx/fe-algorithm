"""
应用配置

使用 pydantic-settings 管理配置
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # 应用配置
    app_env: Literal["development", "production", "testing"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    
    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # JWT 认证
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    
    # LLM 配置
    llm_provider: Literal["openai", "stub"] = "stub"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    
    # 嵌入模型配置
    embedding_provider: Literal["openai", "sentence_transformer", "stub"] = "stub"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 384
    
    # RAG 配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    score_threshold: float = 0.3
    
    # 存储配置
    data_dir: Path = Field(default=Path("./data"))
    index_dir: Path = Field(default=Path("./data/index"))
    upload_dir: Path = Field(default=Path("./data/uploads"))
    
    # 安全配置
    max_input_length: int = 10000
    enable_injection_detection: bool = True
    enable_output_moderation: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


