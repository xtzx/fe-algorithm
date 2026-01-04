# 常用模式

> 配置管理、日志初始化、CLI 框架的最佳实践

## 1. 配置管理

### pydantic-settings 基础

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 基础配置
    app_name: str = Field(default="myapp")
    debug: bool = Field(default=False)

    # 数据库配置
    database_url: str = Field(default="sqlite:///./app.db")
```

### 环境变量映射

```python
# .env 文件
APP_NAME=production-app
DEBUG=false
DATABASE_URL=postgresql://localhost/prod

# 自动映射
settings = Settings()
print(settings.app_name)  # "production-app"
print(settings.debug)  # False
```

### 嵌套配置

```python
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(default="sqlite:///./app.db")
    pool_size: int = Field(default=5)
    echo: bool = Field(default=False)


class Settings(BaseSettings):
    app_name: str = "myapp"
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)


# 使用
settings = Settings()
print(settings.db.url)
print(settings.db.pool_size)
```

### 配置单例

```python
from functools import lru_cache


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 使用
settings = get_settings()
```

### 验证器

```python
from pydantic import field_validator


class Settings(BaseSettings):
    log_level: str = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        return v.upper()
```

## 2. 日志初始化

### 基础配置

```python
import logging


def setup_logging(
    level: str = "INFO",
    format: str = "text",
) -> None:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# 使用
setup_logging(level="DEBUG")
logger = logging.getLogger(__name__)
logger.info("Application started")
```

### 结构化日志

```python
import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


# 使用
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.root.addHandler(handler)
```

### 日志配置模式

```python
# 在应用入口配置一次
def main():
    setup_logging(
        level=settings.log_level,
        format=settings.log_format,
    )

    # 应用逻辑...


# 在各模块获取 logger
logger = logging.getLogger(__name__)
logger.info("Module loaded")
```

### 上下文日志

```python
from logging import LoggerAdapter


class ContextLogger(LoggerAdapter):
    """带上下文的 logger"""

    def process(self, msg, kwargs):
        context = " ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"[{context}] {msg}", kwargs


# 使用
logger = ContextLogger(
    logging.getLogger(__name__),
    {"request_id": "abc123", "user_id": "user1"}
)
logger.info("Processing request")
# 输出: [request_id=abc123 user_id=user1] Processing request
```

## 3. CLI 框架

### argparse 基础

```python
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="My CLI Tool",
    )

    # 全局参数
    parser.add_argument("-v", "--verbose", action="store_true")

    # 子命令
    subparsers = parser.add_subparsers(dest="command")

    # run 命令
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=str)

    args = parser.parse_args()

    if args.command == "run":
        # 执行 run 命令
        pass
```

### 命令函数模式

```python
def cmd_run(args):
    """run 命令处理函数"""
    print(f"Running with config: {args.config}")
    return 0


def cmd_version(args):
    """version 命令处理函数"""
    print("version 1.0.0")
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # 注册命令
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config")
    run_parser.set_defaults(func=cmd_run)

    version_parser = subparsers.add_parser("version")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0
```

### 完整 CLI 模板

```python
import argparse
import sys

from myapp import __version__
from myapp.config import get_settings
from myapp.log import setup_logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myapp",
        description="My Application",
    )

    # 全局选项
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command")

    # ... 添加子命令

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    # 配置日志
    level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=level)

    # 执行命令
    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## 4. 应用入口模式

### 单入口

```python
# src/myapp/__main__.py
from myapp.cli import main

if __name__ == "__main__":
    main()


# 运行方式
# python -m myapp
# myapp (如果配置了 entry_points)
```

### 多入口

```toml
# pyproject.toml
[project.scripts]
myapp = "myapp.cli:main"
myapp-worker = "myapp.worker:main"
myapp-migrate = "myapp.migrate:main"
```

## 5. 错误处理模式

### 自定义异常

```python
class AppError(Exception):
    """应用基础异常"""
    pass


class ConfigError(AppError):
    """配置错误"""
    pass


class DatabaseError(AppError):
    """数据库错误"""
    pass
```

### 异常处理

```python
def main():
    try:
        app.run()
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    return 0
```

## 6. 依赖注入模式

### 简单工厂

```python
from functools import lru_cache


@lru_cache
def get_database():
    """获取数据库连接"""
    settings = get_settings()
    return Database(settings.database_url)


@lru_cache
def get_cache():
    """获取缓存连接"""
    settings = get_settings()
    return Redis(settings.redis_url)
```

### 依赖容器

```python
class Container:
    """简单的依赖容器"""

    _instances: dict = {}

    @classmethod
    def get(cls, key: str, factory):
        if key not in cls._instances:
            cls._instances[key] = factory()
        return cls._instances[key]

    @classmethod
    def reset(cls):
        cls._instances.clear()


# 使用
db = Container.get("database", lambda: Database(settings.database_url))
```

## 7. 测试中的配置

### 覆盖配置

```python
import pytest
from unittest.mock import patch


@pytest.fixture
def test_settings():
    """测试配置"""
    return Settings(
        app_name="test-app",
        debug=True,
        database_url="sqlite:///:memory:",
    )


def test_with_custom_settings(test_settings):
    with patch("myapp.config.get_settings", return_value=test_settings):
        # 测试代码
        pass
```

### 环境变量

```python
import os


@pytest.fixture(autouse=True)
def clean_env():
    """清理环境变量"""
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


def test_config_from_env(clean_env):
    os.environ["APP_NAME"] = "test"
    os.environ["DEBUG"] = "true"

    settings = Settings()
    assert settings.app_name == "test"
    assert settings.debug is True
```

