"""文件工具测试

演示 fixture 使用、monkeypatch、tmp_path。
"""

import json
import pytest
from pathlib import Path

from testing_lab.file_utils import (
    read_json,
    write_json,
    read_text,
    write_text,
    file_exists,
    get_file_size,
    list_directory,
    ensure_directory,
    get_config,
    ConfigManager,
)


# ============================================================
# 使用 tmp_path 测试文件操作
# ============================================================

class TestFileOperations:
    """文件操作测试"""

    def test_write_and_read_json(self, tmp_path: Path):
        """测试 JSON 读写"""
        file_path = tmp_path / "test.json"
        data = {"name": "Alice", "age": 30}

        write_json(file_path, data)
        result = read_json(file_path)

        assert result == data

    def test_write_and_read_text(self, tmp_path: Path):
        """测试文本读写"""
        file_path = tmp_path / "test.txt"
        content = "Hello, World!"

        write_text(file_path, content)
        result = read_text(file_path)

        assert result == content

    def test_read_json_file_not_found(self, tmp_path: Path):
        """测试读取不存在的文件"""
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "nonexistent.json")

    def test_read_json_invalid(self, tmp_path: Path):
        """测试读取无效的 JSON"""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            read_json(file_path)


# ============================================================
# 使用 fixture 的测试
# ============================================================

class TestWithFixtures:
    """使用 fixture 的测试"""

    def test_read_json_from_fixture(self, temp_json_file: Path):
        """使用 conftest.py 中的 temp_json_file fixture"""
        result = read_json(temp_json_file)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_read_text_from_fixture(self, temp_text_file: Path):
        """使用 conftest.py 中的 temp_text_file fixture"""
        result = read_text(temp_text_file)
        assert result == "Hello, World!"


# ============================================================
# 文件系统操作测试
# ============================================================

class TestFileSystem:
    """文件系统操作测试"""

    def test_file_exists(self, temp_text_file: Path):
        assert file_exists(temp_text_file) is True

    def test_file_not_exists(self, tmp_path: Path):
        assert file_exists(tmp_path / "nonexistent.txt") is False

    def test_get_file_size(self, tmp_path: Path):
        file_path = tmp_path / "test.txt"
        content = "12345"  # 5 bytes
        file_path.write_text(content)

        assert get_file_size(file_path) == 5

    def test_get_file_size_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            get_file_size(tmp_path / "nonexistent.txt")

    def test_list_directory(self, tmp_path: Path):
        # 创建一些文件
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "file2.txt").write_text("2")
        (tmp_path / "file3.py").write_text("3")

        all_files = list_directory(tmp_path)
        txt_files = list_directory(tmp_path, "*.txt")

        assert len(all_files) == 3
        assert len(txt_files) == 2

    def test_ensure_directory(self, tmp_path: Path):
        new_dir = tmp_path / "a" / "b" / "c"

        result = ensure_directory(new_dir)

        assert result.exists()
        assert result.is_dir()


# ============================================================
# 使用 monkeypatch 测试环境变量
# ============================================================

class TestConfig:
    """配置测试（使用 monkeypatch）"""

    def test_get_config_default(self, clean_env):
        """测试默认配置"""
        config = get_config()

        assert config["debug"] is False
        assert config["api_key"] == ""
        assert config["log_level"] == "INFO"

    def test_get_config_with_env(self, mock_env_vars):
        """测试从环境变量读取配置"""
        config = get_config()

        assert config["debug"] is True
        assert config["api_key"] == "test-api-key-12345"
        assert config["database_url"] == "sqlite:///:memory:"
        assert config["log_level"] == "DEBUG"

    def test_get_config_custom(self, monkeypatch):
        """自定义环境变量"""
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("API_KEY", "custom-key")

        config = get_config()

        assert config["debug"] is True
        assert config["api_key"] == "custom-key"


# ============================================================
# ConfigManager 测试
# ============================================================

class TestConfigManager:
    """ConfigManager 测试"""

    def test_load_config(self, temp_config_file: Path):
        manager = ConfigManager(temp_config_file)

        config = manager.load()

        assert config["debug"] is True
        assert config["database"]["host"] == "localhost"

    def test_get_config_value(self, temp_config_file: Path):
        manager = ConfigManager(temp_config_file)

        assert manager.get("debug") is True
        assert manager.get("nonexistent", "default") == "default"

    def test_set_config_value(self, temp_config_file: Path):
        manager = ConfigManager(temp_config_file)

        manager.set("new_key", "new_value")

        # 重新加载验证
        manager._config = None
        assert manager.get("new_key") == "new_value"

    def test_save_config(self, tmp_path: Path):
        config_file = tmp_path / "new_config.json"
        config_file.write_text("{}")

        manager = ConfigManager(config_file)
        manager.save({"key": "value"})

        # 验证文件内容
        saved = json.loads(config_file.read_text())
        assert saved["key"] == "value"


# ============================================================
# 使用工厂 Fixture
# ============================================================

class TestWithFactory:
    """使用工厂 fixture 的测试"""

    def test_make_temp_file(self, make_temp_file):
        """使用 make_temp_file 工厂"""
        file1 = make_temp_file("file1.txt", "content1")
        file2 = make_temp_file("file2.txt", "content2")

        assert file1.read_text() == "content1"
        assert file2.read_text() == "content2"

    def test_make_user(self, make_user):
        """使用 make_user 工厂"""
        user1 = make_user("Alice", 30)
        user2 = make_user("Bob", 25, "bob@test.com")

        assert user1["name"] == "Alice"
        assert user1["age"] == 30
        assert user2["email"] == "bob@test.com"

