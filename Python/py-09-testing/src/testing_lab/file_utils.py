"""文件工具模块 - 用于演示 fixture 和 mock"""

import json
import os
from pathlib import Path
from typing import Any


def read_json(file_path: str | Path) -> dict:
    """读取 JSON 文件

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: str | Path, data: dict) -> None:
    """写入 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_text(file_path: str | Path) -> str:
    """读取文本文件"""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def write_text(file_path: str | Path, content: str) -> None:
    """写入文本文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def file_exists(file_path: str | Path) -> bool:
    """检查文件是否存在"""
    return Path(file_path).exists()


def get_file_size(file_path: str | Path) -> int:
    """获取文件大小（字节）

    Raises:
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return path.stat().st_size


def list_directory(dir_path: str | Path, pattern: str = "*") -> list[Path]:
    """列出目录内容"""
    path = Path(dir_path)
    return list(path.glob(pattern))


def ensure_directory(dir_path: str | Path) -> Path:
    """确保目录存在，不存在则创建"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config() -> dict[str, Any]:
    """获取配置（从环境变量）"""
    return {
        "debug": os.environ.get("DEBUG", "false").lower() == "true",
        "api_key": os.environ.get("API_KEY", ""),
        "database_url": os.environ.get("DATABASE_URL", "sqlite:///app.db"),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    }


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._config: dict | None = None

    def load(self) -> dict:
        """加载配置"""
        if self._config is None:
            self._config = read_json(self.config_path)
        return self._config

    def save(self, config: dict) -> None:
        """保存配置"""
        write_json(self.config_path, config)
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        config = self.load()
        return config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        config = self.load()
        config[key] = value
        self.save(config)

