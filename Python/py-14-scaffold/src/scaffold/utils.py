"""
工具函数模块

常用的工具函数集合
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from scaffold.log import get_logger


logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# 装饰器
# =============================================================================
def timer(func: Callable[..., T]) -> Callable[..., T]:
    """
    计时装饰器

    Example:
        @timer
        def slow_function():
            time.sleep(1)
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    重试装饰器

    Example:
        @retry(max_attempts=3, delay=1.0)
        def unstable_operation():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(delay)

            raise last_error  # type: ignore

        return wrapper

    return decorator


# =============================================================================
# 文件操作
# =============================================================================
def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """写入 JSON 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    计算文件哈希

    Args:
        path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)

    Returns:
        哈希值（十六进制）
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# 字符串处理
# =============================================================================
def snake_to_camel(name: str) -> str:
    """snake_case 转 camelCase"""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(name: str) -> str:
    """camelCase 转 snake_case"""
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


# =============================================================================
# 日期时间
# =============================================================================
def now_iso() -> str:
    """当前时间 ISO 格式"""
    return datetime.now().isoformat()


def now_timestamp() -> float:
    """当前时间戳"""
    return time.time()


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# =============================================================================
# 系统命令
# =============================================================================
def run_command(
    command: list[str] | str,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    运行系统命令

    Args:
        command: 命令（列表或字符串）
        cwd: 工作目录
        capture: 是否捕获输出
        check: 是否检查返回码

    Returns:
        CompletedProcess 结果
    """
    if isinstance(command, str):
        command = command.split()

    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=capture,
        text=True,
        check=check,
    )


def get_env(key: str, default: str | None = None) -> str | None:
    """获取环境变量"""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """获取必需的环境变量"""
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Required environment variable not set: {key}")
    return value


# =============================================================================
# 其他工具
# =============================================================================
def chunk_list(lst: list[T], size: int) -> list[list[T]]:
    """将列表分块"""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten(nested: list[list[T]]) -> list[T]:
    """展平嵌套列表"""
    return [item for sublist in nested for item in sublist]


def unique(items: list[T]) -> list[T]:
    """去重（保持顺序）"""
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 测试装饰器
    @timer
    def slow_func():
        time.sleep(0.1)
        return "done"

    print(slow_func())

    # 测试字符串处理
    print(snake_to_camel("hello_world"))  # helloWorld
    print(camel_to_snake("helloWorld"))  # hello_world

    # 测试持续时间格式化
    print(format_duration(0.5))  # 500ms
    print(format_duration(65))  # 1m 5s
    print(format_duration(3700))  # 1h 1m

