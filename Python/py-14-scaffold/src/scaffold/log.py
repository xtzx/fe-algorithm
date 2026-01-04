"""
日志配置模块

提供统一的日志配置：
- 支持文本和 JSON 格式
- 支持不同日志级别
- 支持日志轮转（可选）
- 结构化日志
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal


# =============================================================================
# 日志格式化器
# =============================================================================
class TextFormatter(logging.Formatter):
    """
    文本格式化器

    输出格式：
    2024-01-01 12:00:00 | INFO     | module:function:10 | Message
    """

    # 日志级别颜色（用于终端输出）
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stderr.isatty()
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        # 时间戳
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # 日志级别
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8s}{self.RESET}"
        else:
            level = f"{level:8s}"

        # 位置信息
        location = f"{record.module}:{record.funcName}:{record.lineno}"

        # 消息
        message = record.getMessage()

        # 异常信息
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"{timestamp} | {level} | {location} | {message}"


class JsonFormatter(logging.Formatter):
    """
    JSON 格式化器

    输出 JSON Lines 格式，适合日志聚合系统
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


# =============================================================================
# 日志配置函数
# =============================================================================
def setup_logging(
    level: str | int = "INFO",
    format: Literal["text", "json"] = "text",
    log_file: Path | str | None = None,
    use_colors: bool = True,
) -> None:
    """
    配置日志

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: 日志格式 (text, json)
        log_file: 日志文件路径（可选）
        use_colors: 是否使用颜色（仅 text 格式）

    Example:
        # 基础配置
        setup_logging(level="INFO")

        # JSON 格式（生产环境推荐）
        setup_logging(level="INFO", format="json")

        # 同时输出到文件
        setup_logging(level="DEBUG", log_file="app.log")
    """
    # 获取根 logger
    root_logger = logging.getLogger()

    # 清除现有 handlers
    root_logger.handlers.clear()

    # 设置日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)

    # 创建格式化器
    if format == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter(use_colors=use_colors)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        # 文件日志不使用颜色
        if format == "json":
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(TextFormatter(use_colors=False))
        root_logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    获取 logger

    Args:
        name: logger 名称（通常使用 __name__）

    Returns:
        Logger 实例

    Example:
        logger = get_logger(__name__)
        logger.info("Hello, world!")
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    日志适配器

    支持添加上下文信息

    Example:
        logger = get_context_logger(__name__, user_id="123")
        logger.info("User action")  # 自动包含 user_id
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        # 将 extra 添加到消息中
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra

        # 也可以添加到消息前缀
        if self.extra:
            context = " ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"[{context}] {msg}"

        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """
    获取带上下文的 logger

    Args:
        name: logger 名称
        **context: 上下文信息

    Returns:
        LoggerAdapter 实例

    Example:
        logger = get_context_logger(__name__, request_id="abc123")
        logger.info("Processing request")
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 配置日志
    setup_logging(level="DEBUG", format="text")

    # 获取 logger
    logger = get_logger(__name__)

    # 测试各个级别
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # 带上下文的日志
    ctx_logger = get_context_logger(__name__, request_id="abc123", user_id="user1")
    ctx_logger.info("Processing request")

    # 测试 JSON 格式
    print("\n--- JSON Format ---")
    setup_logging(level="INFO", format="json")
    logger.info("JSON formatted message")

    # 测试异常日志
    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("An error occurred")

