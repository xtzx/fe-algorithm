"""共享 Fixture

这个文件包含所有测试共享的 fixture。
pytest 会自动加载这个文件。
"""

import json
import pytest
from pathlib import Path


# ============================================================
# 基础 Fixture
# ============================================================

@pytest.fixture
def sample_data() -> dict:
    """基本测试数据"""
    return {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
    }


@pytest.fixture
def sample_list() -> list[int]:
    """测试用数字列表"""
    return [1, 2, 3, 4, 5]


# ============================================================
# 文件相关 Fixture
# ============================================================

@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """创建临时 JSON 文件"""
    file_path = tmp_path / "test.json"
    data = {"key": "value", "number": 42}
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    """创建临时文本文件"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """创建临时配置文件"""
    config = {
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
        },
        "features": ["a", "b", "c"],
    }
    file_path = tmp_path / "config.json"
    file_path.write_text(json.dumps(config))
    return file_path


# ============================================================
# 环境相关 Fixture
# ============================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def clean_env(monkeypatch):
    """清除相关环境变量"""
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)


# ============================================================
# 工厂 Fixture
# ============================================================

@pytest.fixture
def make_user():
    """用户工厂"""
    created_users = []

    def _make_user(name: str = "Test User", age: int = 25, email: str = None):
        if email is None:
            email = f"{name.lower().replace(' ', '.')}@example.com"
        user = {"name": name, "age": age, "email": email}
        created_users.append(user)
        return user

    yield _make_user

    # 清理（如果需要）
    created_users.clear()


@pytest.fixture
def make_temp_file(tmp_path: Path):
    """临时文件工厂"""
    created_files = []

    def _make_temp_file(name: str, content: str = "") -> Path:
        file_path = tmp_path / name
        file_path.write_text(content)
        created_files.append(file_path)
        return file_path

    yield _make_temp_file

    # 清理
    for f in created_files:
        if f.exists():
            f.unlink()


# ============================================================
# 标记
# ============================================================

def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "unit: 单元测试")

