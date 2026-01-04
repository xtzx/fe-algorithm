"""工具函数模块"""

import hashlib
import re
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    """将文本转换为 slug

    Args:
        text: 原始文本

    Returns:
        slug 格式的文本
    """
    # 转小写
    text = text.lower()
    # 替换空格
    text = re.sub(r"\s+", "-", text)
    # 移除非字母数字字符
    text = re.sub(r"[^a-z0-9-]", "", text)
    # 移除多余的横线
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def hash_file(path: Path) -> str:
    """计算文件的 MD5 哈希

    Args:
        path: 文件路径

    Returns:
        MD5 哈希值
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def truncate(text: str, length: int = 50, suffix: str = "...") -> str:
    """截断文本

    Args:
        text: 原始文本
        length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if len(text) <= length:
        return text
    return text[: length - len(suffix)] + suffix


def parse_bool(value: Any) -> bool:
    """解析布尔值

    Args:
        value: 任意值

    Returns:
        布尔值
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "on")
    return bool(value)

