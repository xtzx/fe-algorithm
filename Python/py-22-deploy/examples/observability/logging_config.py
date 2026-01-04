"""
结构化日志配置

演示:
- structlog 配置
- JSON 日志格式
- 上下文绑定
- 日志级别
"""

import logging
import sys
from typing import Any

import structlog


def configure_structlog(
    log_level: str = "INFO",
    log_format: str = "json",  # json or console
    service_name: str = "myapp",
):
    """
    配置 structlog
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_format: 输出格式 (json 或 console)
        service_name: 服务名称
    """
    
    # 共享处理器
    shared_processors = [
        # 添加服务名称
        structlog.processors.add_log_level,
        # 添加时间戳
        structlog.processors.TimeStamper(fmt="iso"),
        # 添加调用者信息
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        # 合并 contextvars
        structlog.contextvars.merge_contextvars,
        # 渲染异常
        structlog.processors.format_exc_info,
    ]
    
    # 选择渲染器
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库 logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """获取 logger"""
    return structlog.get_logger(name)


# ==================== 日志上下文 ====================


def bind_context(**kwargs):
    """绑定上下文变量"""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys):
    """解绑上下文变量"""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context():
    """清除所有上下文变量"""
    structlog.contextvars.clear_contextvars()


# ==================== 使用示例 ====================


def demo_logging():
    """演示日志使用"""
    
    # 配置
    configure_structlog(log_level="DEBUG", log_format="console")
    
    logger = get_logger("demo")
    
    # 基本日志
    logger.info("application_started", version="1.0.0")
    logger.debug("debug_message", data={"key": "value"})
    logger.warning("warning_message", warning_code=123)
    logger.error("error_occurred", error_code="ERR001")
    
    # 绑定上下文（会添加到后续所有日志）
    bind_context(request_id="req-123", user_id=42)
    
    logger.info("processing_request", path="/api/users")
    logger.info("database_query", table="users", duration_ms=15)
    
    # 清除上下文
    clear_context()
    
    # 结构化异常日志
    try:
        raise ValueError("Something went wrong")
    except Exception:
        logger.exception("unexpected_error")


if __name__ == "__main__":
    demo_logging()


