"""
共享测试 fixtures
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clean_env():
    """
    清理环境变量

    自动在每个测试前后清理环境变量，
    避免测试之间的相互影响
    """
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def temp_env():
    """
    临时环境变量 fixture

    用于在测试中设置临时环境变量
    """

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)

    return _set_env


@pytest.fixture
def mock_settings():
    """
    Mock 配置 fixture

    用于覆盖配置
    """
    from scaffold.config import Settings

    settings = Settings(
        app_name="test-app",
        app_env="development",
        debug=True,
        log_level="DEBUG",
    )

    with patch("scaffold.config.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """临时目录 fixture"""
    return tmp_path


@pytest.fixture
def sample_env_file(temp_dir: Path) -> Path:
    """创建示例 .env 文件"""
    env_file = temp_dir / ".env"
    env_file.write_text(
        """
APP_NAME=test-from-file
DEBUG=true
LOG_LEVEL=DEBUG
"""
    )
    return env_file

