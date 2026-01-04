"""Config 测试"""

import pytest
from pathlib import Path

from code_counter.config import Config, init_config, show_config


class TestConfig:
    """Config 测试类"""

    def test_default_config(self):
        """测试默认配置"""
        config = Config()

        assert config.default_format == "table"
        assert config.include_hidden is False
        assert config.use_gitignore is True
        assert len(config.exclude) > 0

    def test_load_config_file(self, config_file: Path):
        """测试加载配置文件"""
        config = Config.load(config_path=config_file)

        assert config.default_format == "json"
        assert "node_modules" in config.exclude

    def test_load_nonexistent_file(self, tmp_path: Path):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            Config.load(config_path=tmp_path / "nonexistent.toml")

    def test_search_config(self, tmp_path: Path, config_file: Path):
        """测试搜索配置文件"""
        # 复制配置文件到 tmp_path
        config_path = tmp_path / ".code-counter.toml"
        config_path.write_text(config_file.read_text())

        config = Config.load(search_dir=tmp_path)

        assert config._config_file == config_path

    def test_to_dict(self):
        """测试转换为字典"""
        config = Config()
        data = config.to_dict()

        assert "exclude" in data
        assert "default_format" in data
        assert "use_gitignore" in data

    def test_save_config(self, tmp_path: Path):
        """测试保存配置"""
        config = Config()
        config.default_format = "markdown"

        path = tmp_path / "test.toml"
        config.save(path)

        assert path.exists()
        content = path.read_text()
        assert "markdown" in content

    def test_env_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("CODE_COUNTER_FORMAT", "json")

        config = Config.load()

        assert config.default_format == "json"


class TestInitConfig:
    """init_config 测试"""

    def test_init_config(self, tmp_path: Path, monkeypatch):
        """测试初始化配置"""
        monkeypatch.chdir(tmp_path)

        path = init_config()

        assert path.exists()
        assert path.name == ".code-counter.toml"

    def test_init_config_exists(self, tmp_path: Path, monkeypatch):
        """测试配置已存在"""
        monkeypatch.chdir(tmp_path)

        # 创建已存在的配置
        (tmp_path / ".code-counter.toml").write_text("exists")

        with pytest.raises(FileExistsError):
            init_config()


class TestShowConfig:
    """show_config 测试"""

    def test_show_config(self):
        """测试显示配置"""
        config = Config()
        output = show_config(config)

        assert "当前配置" in output
        assert "默认格式" in output

    def test_show_config_with_file(self, config_file: Path):
        """测试显示已加载的配置"""
        config = Config.load(config_path=config_file)
        output = show_config(config)

        assert str(config_file) in output

