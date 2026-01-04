"""
测试配置模块
"""

import os
import pytest
from unittest.mock import patch

from scaffold.config import Settings, get_settings


class TestSettings:
    """测试 Settings 类"""

    def test_default_values(self):
        """测试默认值"""
        settings = Settings()

        assert settings.app_name == "scaffold"
        assert settings.app_env == "development"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_from_env(self, temp_env):
        """测试从环境变量读取"""
        temp_env(
            APP_NAME="my-app",
            APP_ENV="production",
            DEBUG="true",
            LOG_LEVEL="DEBUG",
        )

        settings = Settings()

        assert settings.app_name == "my-app"
        assert settings.app_env == "production"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_log_level_uppercase(self, temp_env):
        """测试日志级别自动转大写"""
        temp_env(LOG_LEVEL="debug")

        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_is_production(self, temp_env):
        """测试 is_production 属性"""
        temp_env(APP_ENV="production")
        settings = Settings()
        assert settings.is_production is True

        temp_env(APP_ENV="development")
        settings = Settings()
        assert settings.is_production is False

    def test_is_development(self, temp_env):
        """测试 is_development 属性"""
        temp_env(APP_ENV="development")
        settings = Settings()
        assert settings.is_development is True

    def test_model_dump(self):
        """测试序列化"""
        settings = Settings()
        data = settings.model_dump()

        assert "app_name" in data
        assert "app_env" in data
        assert "debug" in data

    def test_model_dump_json(self):
        """测试 JSON 序列化"""
        settings = Settings()
        json_str = settings.model_dump_json()

        assert "app_name" in json_str
        assert "scaffold" in json_str


class TestGetSettings:
    """测试 get_settings 函数"""

    def test_returns_settings(self):
        """测试返回 Settings 实例"""
        # 清除缓存
        get_settings.cache_clear()

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached(self):
        """测试缓存功能"""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

