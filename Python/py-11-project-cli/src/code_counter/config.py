"""配置管理

支持 TOML 配置文件和环境变量。
"""

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 配置文件名
CONFIG_FILE_NAMES = [
    ".code-counter.toml",
    "code-counter.toml",
    ".codecounter.toml",
]

# 默认配置
DEFAULT_CONFIG = {
    "exclude": [
        ".git",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
    ],
    "default_format": "table",
    "include_hidden": False,
    "use_gitignore": True,
}


@dataclass
class Config:
    """配置类"""

    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_CONFIG["exclude"]))
    default_format: str = "table"
    include_hidden: bool = False
    use_gitignore: bool = True
    languages: dict[str, str] = field(default_factory=dict)

    # 配置来源
    _config_file: Path | None = None

    @classmethod
    def load(cls, config_path: Path | None = None, search_dir: Path | None = None) -> "Config":
        """加载配置

        Args:
            config_path: 指定的配置文件路径
            search_dir: 搜索配置文件的目录

        Returns:
            Config 对象
        """
        config = cls()

        # 1. 加载配置文件
        if config_path:
            config._load_file(config_path)
        else:
            config._search_and_load(search_dir or Path.cwd())

        # 2. 环境变量覆盖
        config._load_env()

        return config

    def _search_and_load(self, search_dir: Path) -> None:
        """搜索并加载配置文件"""
        # 向上搜索
        current = search_dir.resolve()
        while current != current.parent:
            for name in CONFIG_FILE_NAMES:
                config_file = current / name
                if config_file.exists():
                    self._load_file(config_file)
                    return
            current = current.parent

        # 检查 home 目录
        home = Path.home()
        for name in CONFIG_FILE_NAMES:
            config_file = home / name
            if config_file.exists():
                self._load_file(config_file)
                return

        logger.debug("No config file found, using defaults")

    def _load_file(self, path: Path) -> None:
        """加载配置文件"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            self._apply_dict(data)
            self._config_file = path
            logger.info(f"Loaded config from {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            raise

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """应用字典配置"""
        if "exclude" in data:
            # 合并而非替换
            self.exclude = list(set(self.exclude + data["exclude"]))

        if "default_format" in data:
            self.default_format = data["default_format"]

        if "include_hidden" in data:
            self.include_hidden = data["include_hidden"]

        if "use_gitignore" in data:
            self.use_gitignore = data["use_gitignore"]

        if "languages" in data:
            self.languages.update(data["languages"])

    def _load_env(self) -> None:
        """从环境变量加载"""
        if "CODE_COUNTER_FORMAT" in os.environ:
            self.default_format = os.environ["CODE_COUNTER_FORMAT"]

        if "CODE_COUNTER_EXCLUDE" in os.environ:
            extra = os.environ["CODE_COUNTER_EXCLUDE"].split(",")
            self.exclude.extend(extra)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "exclude": self.exclude,
            "default_format": self.default_format,
            "include_hidden": self.include_hidden,
            "use_gitignore": self.use_gitignore,
            "languages": self.languages,
        }

    def save(self, path: Path) -> None:
        """保存配置到文件"""
        content = self._to_toml()
        path.write_text(content, encoding="utf-8")
        logger.info(f"Config saved to {path}")

    def _to_toml(self) -> str:
        """转换为 TOML 字符串"""
        lines = [
            "# Code Counter 配置文件",
            "",
            "# 排除的目录/文件模式",
            f'exclude = {self.exclude!r}'.replace("'", '"'),
            "",
            "# 默认输出格式: table, json, markdown",
            f'default_format = "{self.default_format}"',
            "",
            "# 是否包含隐藏文件",
            f"include_hidden = {'true' if self.include_hidden else 'false'}",
            "",
            "# 是否读取 .gitignore",
            f"use_gitignore = {'true' if self.use_gitignore else 'false'}",
            "",
            "# 自定义语言扩展名映射",
            "[languages]",
        ]

        for ext, lang in self.languages.items():
            lines.append(f'"{ext}" = "{lang}"')

        return "\n".join(lines) + "\n"


def init_config(path: Path | None = None) -> Path:
    """初始化配置文件

    Args:
        path: 配置文件路径，默认为当前目录

    Returns:
        创建的配置文件路径
    """
    if path is None:
        path = Path.cwd() / ".code-counter.toml"

    if path.exists():
        raise FileExistsError(f"Config file already exists: {path}")

    config = Config()
    config.save(path)
    return path


def show_config(config: Config) -> str:
    """显示配置信息"""
    lines = [
        "当前配置:",
        f"  配置文件: {config._config_file or '(未加载)'}",
        f"  默认格式: {config.default_format}",
        f"  包含隐藏: {config.include_hidden}",
        f"  使用 gitignore: {config.use_gitignore}",
        f"  排除模式: {len(config.exclude)} 个",
    ]

    for pattern in config.exclude[:5]:
        lines.append(f"    - {pattern}")
    if len(config.exclude) > 5:
        lines.append(f"    ... 还有 {len(config.exclude) - 5} 个")

    return "\n".join(lines)

